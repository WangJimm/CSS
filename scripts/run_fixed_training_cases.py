from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


CS_LOCATIONS = ["42496394", "42490303", "42474000"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a fixed LinUCB/Dijkstra training schedule for one CSS project."
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sumo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--probabilities", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args(argv)


def _optional_float(value):
    return None if value is None else float(value)


def _optional_int(value):
    return None if value is None else int(value)


def _state_snapshot(states):
    return {
        ev: {
            "done": bool(state.done),
            "out": bool(state.out),
            "goal_cs": state.goal_cs,
            "action": _optional_int(state.action),
            "travel_time_seconds": _optional_float(state.travel_time),
            "waiting_time_seconds": _optional_float(state.waiting_time),
            "charging_time_seconds": _optional_float(state.charging_time),
            "decision_count": int(state.decision),
        }
        for ev, state in sorted(states.items())
    }


def _agent_snapshot(agent):
    return {
        "a_inv": agent.a_inv.tolist(),
        "b": agent.b.tolist(),
        "action_counts": agent.action_counts.tolist(),
        "learn_step_counter": int(agent.learn_step_counter),
        "memory_size": len(agent.memory),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    project = args.project.resolve()
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project))

    import config
    import training
    import traci
    import utils_sumo
    from contextual_bandit import ContextualLinUCBAgent
    from env import Env
    from sumolib.net import readNet

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    implementation = sys.modules[training.DRL_train.__module__]
    schedule = [
        training.ScenarioKey(
            scenario_index=0,
            probability_index=probability,
            seed=seed,
        )
        for seed in args.seeds
        for probability in args.probabilities
    ]

    command = [
        str(args.sumo.resolve()),
        "-c",
        str(Path(config.cfg_path).resolve()),
        "--step-length",
        "0.5",
        "--no-step-log",
        "--no-warnings",
    ]
    started_at = time.perf_counter()
    traci.start(command)

    original_builder = implementation.build_training_scenario_schedule
    original_load_sumo = implementation.load_sumo
    original_compress = implementation.compress_route_on_depart
    original_results_output_dir = implementation._results_output_dir
    current = {}
    episodes = []

    try:
        net = readNet(config.net_path, withInternal=True)
        static_pack = utils_sumo.prepare_static(net)
        environment = Env(CS_LOCATIONS, static_pack, net, using_gnn=False)
        agent = ContextualLinUCBAgent(
            environment,
            alpha=1.0,
            ridge=1.0,
            replay_capacity=100_000,
            quadratic_features=True,
            seed=args.random_seed,
            enable_tensorboard=False,
        )

        def finalize_episode():
            if not current:
                return
            states = current.get("states", {})
            episodes.append(
                {
                    "episode": len(episodes),
                    "seed": current["cfg"].rd1,
                    "probability_index": current["cfg"].rd2,
                    "flow_index": current["cfg"].rd3,
                    "vehicle_flow": current["cfg"].veh_flow,
                    "ev_count": len(current["cfg"].evs),
                    "evs": _state_snapshot(states),
                    "agent_after_episode": _agent_snapshot(agent),
                }
            )
            current.clear()

        def fixed_schedule(*, num_episodes, **_kwargs):
            if num_episodes != len(schedule):
                raise ValueError(
                    f"Expected {len(schedule)} fixed episodes, got {num_episodes}"
                )
            return list(schedule)

        def capture_load(*positional, **keywords):
            if not keywords.get("eval", positional[0] if positional else False):
                finalize_episode()
            cfg = original_load_sumo(*positional, **keywords)
            if not keywords.get("eval", positional[0] if positional else False):
                current["cfg"] = cfg
            return cfg

        def capture_states(states, *positional, **keywords):
            result = original_compress(states, *positional, **keywords)
            current["states"] = states
            return result

        implementation.build_training_scenario_schedule = fixed_schedule
        implementation.load_sumo = capture_load
        implementation.compress_route_on_depart = capture_states
        implementation._results_output_dir = lambda: artifact_dir

        training_result = training.DRL_train(
            environment,
            agent,
            agent.memory,
            episodes=len(schedule),
            training_start_episode=0,
            run_evaluation=False,
            using_min_waiting=False,
            route_planning_mode="dijkstra",
            schedule_seed=args.random_seed,
            flow_indices=[0],
            probability_indices=list(args.probabilities),
            seed_start=min(args.seeds),
            seed_end=max(args.seeds),
            save_results=False,
            show_sumo_warnings=False,
            checkpoint_filename=args.checkpoint.resolve(),
        )
        finalize_episode()
        train_rewards = [float(value) for value in training_result[0]]
        for episode, reward in zip(episodes, train_rewards):
            episode["reward"] = reward

        payload = {
            "label": args.label,
            "project": str(project),
            "agent": "linucb",
            "route_planning_mode": "dijkstra",
            "random_seed": args.random_seed,
            "schedule": [vars(key) for key in schedule],
            "episodes": episodes,
            "train_rewards": train_rewards,
            "final_agent": _agent_snapshot(agent),
            "checkpoint": str(args.checkpoint.resolve()),
            "elapsed_seconds": time.perf_counter() - started_at,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"fixed training results saved to {args.output}")
    finally:
        implementation.build_training_scenario_schedule = original_builder
        implementation.load_sumo = original_load_sumo
        implementation.compress_route_on_depart = original_compress
        implementation._results_output_dir = original_results_output_dir
        traci.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
