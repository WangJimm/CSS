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
TRAINING_FLOWS = [1800]
TESTING_FLOWS = [1500, 1800, 2100]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run fixed CSS evaluation cases without using either project entry point."
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--sumo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("training", "testing"), default="testing")
    parser.add_argument("--seeds", type=int, nargs="+", default=[51, 52])
    parser.add_argument("--probabilities", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--flow-index", type=int, help="Compatibility option for one flow.")
    parser.add_argument("--flow-indices", type=int, nargs="+", help="One or more flow indices.")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args(argv)


def _optional_float(value):
    return None if value is None else float(value)


def _state_snapshot(states):
    return {
        ev: {
            "done": bool(state.done),
            "out": bool(state.out),
            "goal_cs": state.goal_cs,
            "travel_time_seconds": _optional_float(state.travel_time),
            "waiting_time_seconds": _optional_float(state.waiting_time),
            "charging_time_seconds": _optional_float(state.charging_time),
            "decision_count": int(state.decision),
        }
        for ev, state in sorted(states.items())
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    project = args.project.resolve()
    sys.path.insert(0, str(project))
    flow_indices = args.flow_indices or [args.flow_index if args.flow_index is not None else 1]

    # These imports intentionally resolve from --project, allowing the same
    # harness to exercise the original and refactored implementations.
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
    cases = []
    traci.start(command)

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
    agent.load_para(str(args.checkpoint.resolve()))

    rollout = training._run_evaluation_rollout
    implementation_module = sys.modules[rollout.__module__]
    original_compress = implementation_module.compress_route_on_depart
    state_holder = {}

    def capture_states(states, *positional, **keywords):
        result = original_compress(states, *positional, **keywords)
        state_holder["states"] = states
        return result

    implementation_module.compress_route_on_depart = capture_states

    try:
        for seed in args.seeds:
            for probability in args.probabilities:
                for flow_index in flow_indices:
                    state_holder.clear()
                    case_started_at = time.perf_counter()
                    if args.dataset == "training":
                        cfg = training.load_sumo(
                            scenario_key=training.ScenarioKey(
                                scenario_index=flow_index,
                                probability_index=probability,
                                seed=seed,
                            ),
                            veh_flows=TRAINING_FLOWS,
                        )
                    else:
                        cfg = training.load_sumo(
                            eval=True,
                            rd1=seed,
                            rd2=probability,
                            rd3=flow_index,
                            veh_flows=TESTING_FLOWS,
                        )
                    aggregate = rollout(
                        environment,
                        agent,
                        cfg,
                        route_planning_mode="dijkstra",
                        rollout_mode="agent",
                    )
                    states = state_holder.get("states", {})
                    cases.append(
                        {
                            "seed": seed,
                            "probability_index": probability,
                            "flow_index": flow_index,
                            "vehicle_flow": cfg.veh_flow,
                            "ev_count": len(cfg.evs),
                            "aggregate": aggregate,
                            "evs": _state_snapshot(states),
                            "elapsed_seconds": time.perf_counter() - case_started_at,
                        }
                    )
    finally:
        implementation_module.compress_route_on_depart = original_compress
        traci.close()

    payload = {
        "label": args.label,
        "project": str(project),
        "agent": "linucb",
        "route_planning_mode": "dijkstra",
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint.resolve()),
        "random_seed": args.random_seed,
        "cases": cases,
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"fixed evaluation results saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
