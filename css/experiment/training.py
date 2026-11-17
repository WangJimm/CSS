# from .graph_utils import *
import json
import hashlib
import os
import sys
import copy
import random
import time
import numpy as np
import torch
import traci
from tqdm import tqdm
from collections import deque
from torch_geometric.loader import DataLoader
from css.experiment.config import (
    RUN_DIR,
    training_veh_flows,
    validation_veh_flows,
    testing_veh_flows,
    using_env,
    cfg_path,
    EVState,
    MAX_DECISIONS_PER_EV,
    ROUTE_FAILURE_HORIZON_SEC,
    DECISION_DIST_THRESHOLD,
    EMA_feature,
    ROUTE_PLANNING_MODE,
    TRAINING_SEED_END,
    TRAINING_SEED_START,
)
from traci.exceptions import FatalTraCIError, TraCIException
from css.experiment.departures import compress_route_on_depart
from css.graph import build_data_runtime, dynamic_node_feature_update
from css.sumo import (
    action_mask_dqn,
    apply_full_route,
    apply_next_junction,
    candidate_next_junctions,
    count_evs_in_rou,
    find_sumo_route_to_node,
    find_tls_and_link_index_for_movement,
    get_evs_in_rou,
    remaining_routes_len,
    reset_signal_topology_cache,
    time_to_green_and_remaining_green,
)
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

from css.experiment.scenarios import (
    ScenarioKey,
    scenario_keys_for_split,
    SumoLoadConfig,
    _default_flow_indices,
    _make_json_safe,
    _resolve_veh_flow,
    _results_output_dir,
    build_training_scenario_schedule,
    choose_training_scenario_key,
    load_higher_eval_sumo,
    load_sumo,
    save_training_results,
)

from css.experiment.planning import (
    build_candidate_action_features,
    candidate_lane_metrics,
    check_arrived,
    first_plan,
    oneHot,
    resolve_movement_out_lanes,
    route_planning,
    route_planning_dijkstra,
    second_plan,
)


def _settle_policy_failures(states, now, horizon_sec=ROUTE_FAILURE_HORIZON_SEC):
    """Assign a conservative, explicit cost to EVs not completed by the policy."""
    failures = []
    for ev, st in states.items():
        if st.travel_time is not None and st.waiting_time is not None and st.charging_time is not None:
            continue

        if st.decision >= MAX_DECISIONS_PER_EV:
            reason = "max_decisions"
        elif st.depart_time is None:
            reason = "not_departed_before_rollout_end"
        else:
            reason = "simulation_ended_before_arrival"

        observed_travel_sec = (
            max(0.0, float(now) - float(st.depart_time))
            if st.depart_time is not None
            else 0.0
        )
        existing_travel_sec = float(st.travel_time) if st.travel_time is not None else 0.0
        penalized_travel_sec = max(
            existing_travel_sec,
            observed_travel_sec,
            float(horizon_sec),
        )
        st.travel_time = penalized_travel_sec
        st.waiting_time = 0.0 if st.waiting_time is None else float(st.waiting_time)
        st.charging_time = 0.0 if st.charging_time is None else float(st.charging_time)
        st.out = True
        st.failure_reason = reason
        failures.append(
            {
                "ev": ev,
                "reason": reason,
                "decisions": int(st.decision),
                "goal_cs": st.goal_cs,
                "observed_travel_sec": observed_travel_sec,
                "penalized_travel_sec": penalized_travel_sec,
            }
        )
    return failures

def DRL_train(
    env,
    agent,
    replay_buffer,
    episodes = 500,
    training_start_episode: int = 50,
    run_evaluation: bool = True,
    using_min_waiting = False,
    route_planning_mode = "dqngnn",
    schedule_seed: int = 42,
    flow_indices: List[int] | None = None,
    validation_flow_indices: List[int] | None = None,
    probability_indices: List[int] | None = None,
    seed_start: int = TRAINING_SEED_START,
    seed_end: int = TRAINING_SEED_END,
    save_results: bool = True,
    results_filename: str | None = None,
    show_sumo_warnings: bool = False,
    checkpoint_filename: str | Path | None = None,
    environment=None,
):
    """
    using_min_waiting: 若True, 會增加 min waiting time baseline 的模擬

    """
    if environment is None:
        from css.experiment.environments import resolve_environment
        environment = resolve_environment()
    train_flows = list(environment.training_flows) if environment is not None else training_veh_flows
    valid_flows = list(environment.validation_flows) if environment is not None else validation_veh_flows
    validation_split = "validation" if environment is not None and environment.manifest_based else "testing"
    if checkpoint_filename is None and environment is not None and environment.manifest_based:
        checkpoint_filename = environment.model_dir / f"{type(agent).__name__}_route_{route_planning_mode}.pt"
    if checkpoint_filename is not None:
        Path(checkpoint_filename).parent.mkdir(parents=True, exist_ok=True)
    max_eat = 0
    best_eval_cost = float('-inf')
    show = True
    """ 數值陣列存放區 """
    total_reward = []
    total_eval_costs = []
    total_random_rewards = []
    total_eval_min_waiting_rewards = []
    total_shortest_rewards = []
    critic_losses = []
    actor_losses = []
    alpha_losses = []
    entropys = []
    alphas = []

    eval_fre = max(1, episodes // 10) if run_evaluation else None
    scenario_schedule = build_training_scenario_schedule(
        num_episodes=episodes,
        schedule_seed=schedule_seed,
        veh_flows=train_flows,
        flow_indices=flow_indices,
        probability_indices=probability_indices,
        seed_start=seed_start,
        seed_end=seed_end,
        environment=environment,
    )
    episode_records = []
    evaluation_records = []
    best_data = None
    base_data_1 = None
    base_data_2 = None
    best_evaluation_detail = None
    training_buffer = agent.memory if hasattr(agent, "memory") else replay_buffer
    required_buffer_size = max(1, int(getattr(agent, "batch_size", 1)))
    sumo_error_log = _results_output_dir(environment) / 'sumo_training_errors.log'
    for epi_i in tqdm(range(episodes)):

        episode_reward = 0
        min_waiting_reward = 0
        scenario_key = scenario_schedule[epi_i]
        cfg = load_sumo(eval = False, scenario_key = scenario_key, environment=environment) # cfg相關去看SumoLoadConfig
        load_args = list(cfg.load_args)
        if show_sumo_warnings:
            load_args = [arg for arg in load_args if arg != '--no-warnings']
        elif '--no-warnings' not in load_args:
            load_args.append('--no-warnings')
        load_args.extend(['--error-log', str(sumo_error_log)])
        try:
            reset_signal_topology_cache()
            traci.load(load_args)
        except (FatalTraCIError, TraCIException, OSError) as exc:
            raise RuntimeError(
                f'SUMO failed to load training episode {epi_i}; '
                f'seed={cfg.rd1}, probability={cfg.rd2}, flow={cfg.veh_flow}. '
                f'Inspect {sumo_error_log}. Original error: {type(exc).__name__}: {exc}'
            ) from exc
        # print(f"using ev_flow {cfg.ev_routes}")
        # print(f"ev power {cfg.ev_power}")
        env.episode_reset(cfg.evs, cfg.ev_power)
        # sys.exit("ckp1")


        states = {ev: EVState(n_nodes=len(env.nid2idx)) for ev in cfg.evs}
        for ev in cfg.evs:
            states[ev].power = cfg.ev_power[ev]
        half_life_sec = 5
        travel_time = EMA_feature(half_life_sec)
        MeanSpeed = EMA_feature(half_life_sec)
        Occupancy = EMA_feature(half_life_sec)
        

        decided = {ev: False for ev in states} ## 給dijkstra用的
        sumo_failure = None
        simulation_step_count = 0
        # print("ckp 1")
        while True:
            try:
                remaining_ev = traci.simulation.getMinExpectedNumber()
            except (FatalTraCIError, TraCIException, OSError) as exc:
                sumo_failure = ('getMinExpectedNumber', exc)
                break
            if remaining_ev <= 0:
                break
            try:
                traci.simulationStep()
                simulation_step_count += 1
            except (FatalTraCIError, TraCIException, OSError) as exc:
                sumo_failure = ('simulationStep', exc)
                break
            

            compress_route_on_depart(
                states,
                env,
                agent,
                travel_time,
                MeanSpeed,
                Occupancy,
                # Both incremental and full-route planners start from the EV's
                # actual departure edge. Dijkstra replaces this one-edge route
                # with its cached complete route later in the same step.
                truncate_departure_route=True,
            )
            env.set_route_planning_agent(route_planning_mode)
            if route_planning_mode in {"dqngnn", "dqn", "edqn"}:
                route_planning(states, env, travel_time, MeanSpeed, Occupancy)        ## 開始task 2, route planning
            elif route_planning_mode == "dijkstra":
                route_planning_dijkstra(states, env, decided)                           ## task2, using dijkstra
            check_arrived(states, env)                                              ### 檢查是否抵達

            if all(st.done or st.out or st.decision >= MAX_DECISIONS_PER_EV for ev, st in states.items()):
                break
        # sys.exit("ckp2")
        incomplete_evs = [
            ev
            for ev, st in states.items()
            if st.travel_time is None or st.waiting_time is None or st.charging_time is None
        ]
        if sumo_failure is not None:
            operation, original_error = sumo_failure
            examples = '; '.join(
                f'{ev}(goal={states[ev].goal_cs}, decisions={states[ev].decision}, '
                f'last_edges={states[ev].routes[-3:]})'
                for ev in incomplete_evs[:10]
            ) or 'none'
            raise RuntimeError(
                f'SUMO aborted training episode {epi_i} during {operation} after '
                f'{simulation_step_count} steps; seed={cfg.rd1}, probability={cfg.rd2}, '
                f'flow={cfg.veh_flow}, unfinished_evs={len(incomplete_evs)} '
                f'(examples: {examples}). No incomplete reward was added to the agent. '
                f'Inspect {sumo_error_log}. Original error: '
                f'{type(original_error).__name__}: {original_error}'
            ) from original_error
        route_failures = _settle_policy_failures(
            states,
            traci.simulation.getTime(),
        )
        if route_failures:
            examples = '; '.join(
                f'{item["ev"]}(goal={item["goal_cs"]}, decisions={item["decisions"]}, '
                f'reason={item["reason"]})'
                for item in route_failures[:10]
            )
            print(
                f'[ROUTE POLICY FAILURE] Training episode {epi_i} assigned the '
                f'{ROUTE_FAILURE_HORIZON_SEC / 60:.1f}-minute failure horizon to '
                f'{len(route_failures)} EV(s): {examples}'
            )
        for ev, st in states.items():
            ### push(state, action, reward, next state, done)
            ### 這邊因為每次都是done = True, next state就隨便放
            reward = -(st.travel_time + st.waiting_time + st.charging_time)/60 #代表的是分鐘
            episode_reward += reward
            if st.sacd_state is None or st.action is None:
                continue
            terminal_next_state = None if hasattr(agent, "uses_graph_state") and agent.uses_graph_state else np.zeros_like(np.asarray(st.sacd_state, dtype=np.float32))
            training_buffer.push(st.sacd_state, [st.action], [reward/30] , terminal_next_state, [float(True)]) 
            if epi_i >= training_start_episode and len(training_buffer) >= required_buffer_size:
                agent.update_parameters(training_buffer)
                if show:
                    print(f"start learning..")
                    show = False


        total_reward.append(episode_reward)
        episode_records.append(
            {
                "episode": epi_i,
                "reward": episode_reward,
                "scenario_index": cfg.rd3,
                "scenario_name": cfg.veh_flow,
                "probability_index": cfg.rd2,
                "seed": cfg.rd1,
                "ev_count": len(cfg.evs),
                "route_planning_mode": route_planning_mode,
                "route_file": cfg.ev_routes,
                "general_file": cfg.general,
                "route_failure_count": len(route_failures),
                "route_failures": route_failures,
            }
        )
        # """
        # cs navigation using random
        # 在compress_route_on_depart新增一個using_random = True
        # """
        
        # if using_random:
        #     # print("ckp 2")
        #     traci.load(load_args)
        #     env.episode_reset(cfg.evs, cfg.ev_power)
        #     states_r = {ev: EVState(n_nodes=len(env.nid2idx)) for ev in cfg.evs}
        #     for ev in cfg.evs:
        #         states_r[ev].power = cfg.ev_power[ev]
        #     decided_r = {ev: False for ev in states_r}
        #     while True:
        #         try:
        #             remaining_ev = traci.simulation.getMinExpectedNumber()
        #         except (FatalTraCIError, TraCIException, OSError):
        #             print("[Warning] SUMO closed during getMinExpectedNumber()")
        #             break
        #         if remaining_ev <= 0:
        #             break
        #         try:
        #             traci.simulationStep()
        #         except (FatalTraCIError, TraCIException, OSError):
        #             print("[Warning] SUMO closed during simulationStep()")
        #             break

        #         compress_route_on_depart(states_r, env, agent, using_random=using_random)         # 這邊處理EV route，同時做cs選擇 action 1
        #         if route_planning_mode == "dqngnn":
        #             route_planning(states, env, travel_time, MeanSpeed, Occupancy)        ## 開始task 2, route planning
        #         elif route_planning_mode == "dijkstra":
        #             route_planning_dijkstra(states, env, decided)                           ## task2, using dijkstra
        #         check_arrived(states_r, env)                                              ### 檢查是否抵達

        #         if all(st.done or st.out or st.decision >= MAX_DECISIONS_PER_EV for ev, st in states_r.items()):
        #             break
        #     for ev, st in states_r.items():
        #         reward = -(st.travel_time + st.waiting_time + st.charging_time)/60 #代表的是分鐘
        #         random_reward += reward
        #     total_random_rewards.append(random_reward)


        """
        evaluation
        """
        if run_evaluation and (epi_i) % eval_fre == 0:
            min_waiting_cost = None
            shortest_cost = None
            eval_data_bag = None
            min_waiting_data_bag = None
            shortest_data_bag = None
            eval_detail = None
            if using_min_waiting:
                eval_cost, min_waiting_cost, shortest_cost, eval_data_bag, min_waiting_data_bag, shortest_data_bag, eval_detail = evaluation(
                    env,
                    agent,
                    using_min_waiting = using_min_waiting,
                    using_shortest = True,
                    route_planning_mode = route_planning_mode,
                    return_details = True,
                    veh_flows = valid_flows,
                    flow_indices = validation_flow_indices,
                    environment=environment,
                    split=validation_split,
                )
                total_eval_costs.append(eval_cost)
                total_eval_min_waiting_rewards.append(min_waiting_cost)
                total_shortest_rewards.append(shortest_cost)


            else:
                eval_cost, _, _, eval_data_bag, min_waiting_data_bag, shortest_data_bag, eval_detail = evaluation(
                    env,
                    agent,
                    route_planning_mode = route_planning_mode,
                    return_details = True,
                    veh_flows = valid_flows,
                    flow_indices = validation_flow_indices,
                    environment=environment,
                    split=validation_split,
                )
                total_eval_costs.append(eval_cost)

            evaluation_records.append(
                {
                    "episode": epi_i,
                    "eval_cost": eval_cost,
                    "min_waiting_eval_cost": min_waiting_cost,
                    "shortest_eval_cost": shortest_cost,
                    "eval_breakdown": eval_data_bag,
                    "min_waiting_breakdown": min_waiting_data_bag,
                    "shortest_breakdown": shortest_data_bag,
                    "route_planning_mode": route_planning_mode,
                    "detail": eval_detail,
                }
            )

            if eval_cost > best_eval_cost:
                if checkpoint_filename is None:
                    agent.save_para()
                else:
                    agent.save_para(checkpoint_filename)
                best_eval_cost = eval_cost
                best_data = eval_data_bag
                base_data_1 = min_waiting_data_bag
                base_data_2 = shortest_data_bag
                best_evaluation_detail = eval_detail

    if not run_evaluation:
        if checkpoint_filename is None:
            agent.save_para()
        else:
            agent.save_para(checkpoint_filename)
    agent.writer.close()
    if save_results:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if results_filename is None:
            results_filename = f"training_results_{stamp}.json"
        output_path = _results_output_dir(environment) / results_filename
        payload = {
            "summary": {
                "environment": environment.name if environment is not None else using_env,
                "data_root": str(environment.root if environment is not None else RUN_DIR),
                "validation_split": validation_split,
                "episodes": episodes,
                "run_evaluation": run_evaluation,
                "using_min_waiting": using_min_waiting,
                "route_planning_mode": route_planning_mode,
                "training_start_episode": training_start_episode,
                "schedule_seed": schedule_seed,
                "training_veh_flows": train_flows,
                "validation_veh_flows": valid_flows,
                "training_flow_indices": flow_indices or _default_flow_indices(train_flows),
                "validation_flow_indices": validation_flow_indices or _default_flow_indices(valid_flows),
                "probability_indices": probability_indices or [1, 2],
                "seed_start": seed_start,
                "seed_end": seed_end,
                "numeric_seed_range_applies": not environment.manifest_based,
                "scenario_pool_source": str(environment.root / "scenario_manifest.csv") if environment.manifest_based else "legacy numeric ranges",
                "scenario_manifest_sha256": hashlib.sha256((environment.root / "scenario_manifest.csv").read_bytes()).hexdigest() if environment.manifest_based else None,
                "eval_frequency": eval_fre,
                "best_eval_cost": best_eval_cost if run_evaluation else None,
            },
            "train_rewards": total_reward,
            "eval_rewards": total_eval_costs,
            "min_waiting_eval_rewards": total_eval_min_waiting_rewards,
            "shortest_eval_rewards": total_shortest_rewards,
            "best_data": best_data,
            "min_waiting_best_data": base_data_1,
            "shortest_best_data": base_data_2,
            "best_evaluation_detail": best_evaluation_detail,
            "scenario_schedule": [vars(k) for k in scenario_schedule],
            "episode_records": episode_records,
            "evaluation_records": evaluation_records,
        }
        save_training_results(payload, output_path)
    return total_reward, total_eval_costs, total_eval_min_waiting_rewards, total_shortest_rewards, best_data, base_data_1, base_data_2

def _empty_case_metric_store():
    return {
        "cases": [],
        "travel_cost_minutes": [],
        "waiting_cost_minutes": [],
        "travel_waiting_total_minutes": [],
        "route_failure_count": 0,
    }


def _append_case_metric(
    store,
    scenario_name,
    cfg,
    travel_cost_minutes,
    waiting_cost_minutes,
    route_failure_count=0,
    route_failures=None,
):
    if scenario_name not in store:
        store[scenario_name] = _empty_case_metric_store()
    total_cost_minutes = travel_cost_minutes + waiting_cost_minutes
    store[scenario_name]["travel_cost_minutes"].append(float(travel_cost_minutes))
    store[scenario_name]["waiting_cost_minutes"].append(float(waiting_cost_minutes))
    store[scenario_name]["travel_waiting_total_minutes"].append(float(total_cost_minutes))
    store[scenario_name]["route_failure_count"] += int(route_failure_count)
    store[scenario_name]["cases"].append(
        {
            "seed": int(cfg.rd1),
            "probability_index": int(cfg.rd2),
            "scenario_index": int(cfg.rd3),
            "scenario_name": scenario_name,
            "travel_cost_minutes": float(travel_cost_minutes),
            "waiting_cost_minutes": float(waiting_cost_minutes),
            "travel_waiting_total_minutes": float(total_cost_minutes),
            "route_failure_count": int(route_failure_count),
            "route_failures": list(route_failures or []),
        }
    )


def _summarize_metric_values(values):
    if not values:
        return {"sum": 0.0, "mean": 0.0, "std": 0.0, "values": []}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "sum": float(arr.sum()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "values": [float(v) for v in arr.tolist()],
    }


def _summarize_method_metrics(store):
    per_scenario = {}
    overall_travel = []
    overall_waiting = []
    overall_total = []
    overall_cases = []
    for scenario_name, bucket in store.items():
        per_scenario[scenario_name] = {
            "num_cases": len(bucket["cases"]),
            "travel_cost_minutes": _summarize_metric_values(bucket["travel_cost_minutes"]),
            "waiting_cost_minutes": _summarize_metric_values(bucket["waiting_cost_minutes"]),
            "travel_waiting_total_minutes": _summarize_metric_values(bucket["travel_waiting_total_minutes"]),
            "route_failure_count": int(bucket["route_failure_count"]),
            "cases": bucket["cases"],
        }
        overall_travel.extend(bucket["travel_cost_minutes"])
        overall_waiting.extend(bucket["waiting_cost_minutes"])
        overall_total.extend(bucket["travel_waiting_total_minutes"])
        overall_cases.extend(bucket["cases"])

    return {
        "per_scenario": per_scenario,
        "overall": {
            "num_cases": len(overall_cases),
            "travel_cost_minutes": _summarize_metric_values(overall_travel),
            "waiting_cost_minutes": _summarize_metric_values(overall_waiting),
            "travel_waiting_total_minutes": _summarize_metric_values(overall_total),
            "route_failure_count": int(sum(case["route_failure_count"] for case in overall_cases)),
            "cases": overall_cases,
        },
    }


def _run_evaluation_rollout(env, agent, cfg, route_planning_mode, rollout_mode = "agent"):
    reset_signal_topology_cache()
    traci.load(cfg.load_args)
    env.episode_reset(cfg.evs, cfg.ev_power)

    states = {ev: EVState(n_nodes=len(env.nid2idx)) for ev in cfg.evs}
    for ev in cfg.evs:
        states[ev].power = cfg.ev_power[ev]

    half_life_sec = 5
    travel_time = EMA_feature(half_life_sec)
    MeanSpeed = EMA_feature(half_life_sec)
    Occupancy = EMA_feature(half_life_sec)
    decided = {ev: False for ev in states}
    sumo_failure = None

    while True:
        try:
            remaining_ev = traci.simulation.getMinExpectedNumber()
        except (FatalTraCIError, TraCIException, OSError) as exc:
            sumo_failure = ("getMinExpectedNumber", exc)
            break
        if remaining_ev <= 0:
            break
        try:
            traci.simulationStep()
        except (FatalTraCIError, TraCIException, OSError) as exc:
            sumo_failure = ("simulationStep", exc)
            break

        if rollout_mode == "agent":
            compress_route_on_depart(
                states,
                env,
                agent,
                travel_time,
                MeanSpeed,
                Occupancy,
                eval=True,
                truncate_departure_route=True,
            )
        elif rollout_mode == "min_waiting":
            compress_route_on_depart(
                states,
                env,
                agent,
                travel_time,
                MeanSpeed,
                Occupancy,
                eval=True,
                using_min_waiting=True,
                truncate_departure_route=True,
            )
        elif rollout_mode == "shortest":
            compress_route_on_depart(
                states,
                env,
                agent,
                travel_time,
                MeanSpeed,
                Occupancy,
                eval=True,
                using_shortest=True,
                truncate_departure_route=True,
            )
        else:
            raise ValueError(f"Unknown rollout mode: {rollout_mode}")

        env.set_route_planning_agent(route_planning_mode)
        if route_planning_mode in {"dqngnn", "dqn", "edqn"}:
            route_planning(states, env, travel_time, MeanSpeed, Occupancy)
        elif route_planning_mode == "dijkstra":
            route_planning_dijkstra(states, env, decided)

        check_arrived(states, env)
        if all(st.done or st.out or st.decision >= MAX_DECISIONS_PER_EV for ev, st in states.items()):
            break

    if sumo_failure is not None:
        operation, original_error = sumo_failure
        raise RuntimeError(
            f'SUMO aborted {rollout_mode} evaluation during {operation}; '
            f'seed={cfg.rd1}, probability={cfg.rd2}, flow={cfg.veh_flow}. '
            f'Original error: {type(original_error).__name__}: {original_error}'
        ) from original_error

    route_failures = _settle_policy_failures(states, traci.simulation.getTime())
    if route_failures:
        examples = '; '.join(
            f'{item["ev"]}(goal={item["goal_cs"]}, decisions={item["decisions"]}, '
            f'reason={item["reason"]})'
            for item in route_failures[:10]
        )
        print(
            f'[ROUTE POLICY FAILURE] {rollout_mode} evaluation assigned the '
            f'{ROUTE_FAILURE_HORIZON_SEC / 60:.1f}-minute failure horizon to '
            f'{len(route_failures)} EV(s): {examples}'
        )

    total_reward = 0.0
    total_traveling_time = 0.0
    total_waiting_time = 0.0
    total_charging_time = 0.0
    travel_cost_minutes = 0.0
    waiting_cost_minutes = 0.0
    for ev, st in states.items():
        total_traveling_time += -(st.travel_time)/60
        total_waiting_time += -(st.waiting_time)/60
        total_charging_time += -(st.charging_time)/60
        travel_cost_minutes += (st.travel_time)/60
        waiting_cost_minutes += (st.waiting_time)/60
        reward = -(st.travel_time + st.waiting_time + st.charging_time)/60
        total_reward += reward

    return {
        "total_reward": float(total_reward),
        "breakdown": (
            float(total_traveling_time),
            float(total_waiting_time),
            float(total_charging_time),
        ),
        "travel_cost_minutes": float(travel_cost_minutes),
        "waiting_cost_minutes": float(waiting_cost_minutes),
        "travel_waiting_total_minutes": float(travel_cost_minutes + waiting_cost_minutes),
        "route_failure_count": len(route_failures),
        "route_failures": route_failures,
    }


def evaluation(
    env,
    agent,
    using_min_waiting = False,
    using_shortest = False,
    route_planning_mode = "dqngnn",
    return_details = False,
    veh_flows: List[str | int] | None = None,
    flow_indices: List[int] | None = None,
    environment=None,
    split="testing",
):
    if split not in ("validation", "testing"):
        raise ValueError("Evaluation split must be validation or testing")
    veh_flows = veh_flows or (list(environment.flows(split)) if environment is not None else testing_veh_flows)
    flow_indices = flow_indices or _default_flow_indices(veh_flows)
    total_eval_reward = 0.0
    total_traveling_time = 0.0
    total_waiting_time = 0.0
    total_charging_time = 0.0

    min_waiting_total_traveling_time = 0.0
    min_waiting_total_waiting_time = 0.0
    min_waiting_total_charging_time = 0.0
    total_eval_min_waiting_reward = 0.0

    shortest_total_eval_reward = 0.0
    shortest_total_traveling_time = 0.0
    shortest_total_waiting_time = 0.0
    shortest_total_charging_time = 0.0

    detail_store = {
        "agent": {},
        "min_waiting": {},
        "shortest": {},
    }

    if environment is None:
        from css.experiment.environments import resolve_environment
        selected_environment = resolve_environment()
    else:
        selected_environment = environment
    keys = scenario_keys_for_split(selected_environment, split, veh_flows, flow_indices)
    for key in tqdm(keys, unit="scenario"):
        cfg = load_sumo(eval=True, rd1=key.seed, rd2=key.probability_index, rd3=key.scenario_index,
                        scenario_key=key, veh_flows=veh_flows, environment=environment, split=split)
        scenario_name = cfg.veh_flow

        agent_result = _run_evaluation_rollout(env, agent, cfg, route_planning_mode, rollout_mode = "agent")
        total_eval_reward += agent_result["total_reward"]
        total_traveling_time += agent_result["breakdown"][0]
        total_waiting_time += agent_result["breakdown"][1]
        total_charging_time += agent_result["breakdown"][2]
        _append_case_metric(
            detail_store["agent"],
            scenario_name,
            cfg,
            agent_result["travel_cost_minutes"],
            agent_result["waiting_cost_minutes"],
            agent_result["route_failure_count"],
            agent_result["route_failures"],
        )

        if using_min_waiting:
            min_waiting_result = _run_evaluation_rollout(env, agent, cfg, route_planning_mode, rollout_mode = "min_waiting")
            total_eval_min_waiting_reward += min_waiting_result["total_reward"]
            min_waiting_total_traveling_time += min_waiting_result["breakdown"][0]
            min_waiting_total_waiting_time += min_waiting_result["breakdown"][1]
            min_waiting_total_charging_time += min_waiting_result["breakdown"][2]
            _append_case_metric(
                detail_store["min_waiting"],
                scenario_name,
                cfg,
                min_waiting_result["travel_cost_minutes"],
                min_waiting_result["waiting_cost_minutes"],
                min_waiting_result["route_failure_count"],
                min_waiting_result["route_failures"],
            )

        if using_shortest:
            shortest_result = _run_evaluation_rollout(env, agent, cfg, route_planning_mode, rollout_mode = "shortest")
            shortest_total_eval_reward += shortest_result["total_reward"]
            shortest_total_traveling_time += shortest_result["breakdown"][0]
            shortest_total_waiting_time += shortest_result["breakdown"][1]
            shortest_total_charging_time += shortest_result["breakdown"][2]
            _append_case_metric(
                detail_store["shortest"],
                scenario_name,
                cfg,
                shortest_result["travel_cost_minutes"],
                shortest_result["waiting_cost_minutes"],
                shortest_result["route_failure_count"],
                shortest_result["route_failures"],
            )

    result = (
        total_eval_reward,
        total_eval_min_waiting_reward,
        shortest_total_eval_reward,
        (total_traveling_time, total_waiting_time, total_charging_time),
        (min_waiting_total_traveling_time, min_waiting_total_waiting_time, min_waiting_total_charging_time),
        (shortest_total_traveling_time, shortest_total_waiting_time, shortest_total_charging_time),
    )
    if not return_details:
        return result

    detail = {
        "metric_definition": {
            "travel_cost_minutes": "Positive travel-time cost in minutes per testing case.",
            "waiting_cost_minutes": "Positive waiting-time cost in minutes per testing case.",
            "travel_waiting_total_minutes": "Positive total cost in minutes per testing case, computed as travel + waiting and excluding charging time.",
            "route_failure_count": "Number of EVs that did not finish under the route policy. Each receives a 50-minute censored travel cost.",
        },
        "evaluation_scope": {
            "probability_indices": [1, 2],
            "scenario_indices": flow_indices,
            "scenario_names": [_resolve_veh_flow(veh_flows, rd3) for rd3 in flow_indices],
            "seed_range": [min(k.seed for k in keys), max(k.seed for k in keys)],
            "seeds": sorted({k.seed for k in keys}),
            "split": split,
            "environment": selected_environment.name,
            "scenario_count": len(keys),
            "data_root": str(selected_environment.root),
            "general_data_dir": f"{split}_data",
            "power_data_dir": f"{split}_data_power",
        },
        "methods": {
            "agent": _summarize_method_metrics(detail_store["agent"]),
            "min_waiting": _summarize_method_metrics(detail_store["min_waiting"]) if using_min_waiting else None,
            "shortest": _summarize_method_metrics(detail_store["shortest"]) if using_shortest else None,
        },
    }
    return (*result, detail)


def higher_evaluation(
    env,
    agent,
    using_min_waiting = False,
    using_shortest = False,
    route_planning_mode = "dqngnn",
    return_details = False,
    veh_flows: List[str | int] | None = None,
    flow_indices: List[int] | None = None,
    environment=None,
):
    veh_flows = veh_flows or (list(environment.testing_flows) if environment is not None else testing_veh_flows)
    flow_indices = flow_indices or _default_flow_indices(veh_flows)
    total_eval_reward = 0.0
    total_traveling_time = 0.0
    total_waiting_time = 0.0
    total_charging_time = 0.0

    min_waiting_total_traveling_time = 0.0
    min_waiting_total_waiting_time = 0.0
    min_waiting_total_charging_time = 0.0
    total_eval_min_waiting_reward = 0.0

    shortest_total_eval_reward = 0.0
    shortest_total_traveling_time = 0.0
    shortest_total_waiting_time = 0.0
    shortest_total_charging_time = 0.0

    detail_store = {
        "agent": {},
        "min_waiting": {},
        "shortest": {},
    }

    for rd1 in range(1, 11):
        for rd2 in [4, 6]:
            for rd3 in flow_indices:
                cfg = load_higher_eval_sumo(rd1 = rd1, rd2 = rd2, rd3 = rd3, veh_flows = veh_flows, environment=environment)
                scenario_name = cfg.veh_flow

                agent_result = _run_evaluation_rollout(env, agent, cfg, route_planning_mode, rollout_mode = "agent")
                total_eval_reward += agent_result["total_reward"]
                total_traveling_time += agent_result["breakdown"][0]
                total_waiting_time += agent_result["breakdown"][1]
                total_charging_time += agent_result["breakdown"][2]
                _append_case_metric(
                    detail_store["agent"],
                    scenario_name,
                    cfg,
                    agent_result["travel_cost_minutes"],
                    agent_result["waiting_cost_minutes"],
                    agent_result["route_failure_count"],
                    agent_result["route_failures"],
                )

                if using_min_waiting:
                    min_waiting_result = _run_evaluation_rollout(env, agent, cfg, route_planning_mode, rollout_mode = "min_waiting")
                    total_eval_min_waiting_reward += min_waiting_result["total_reward"]
                    min_waiting_total_traveling_time += min_waiting_result["breakdown"][0]
                    min_waiting_total_waiting_time += min_waiting_result["breakdown"][1]
                    min_waiting_total_charging_time += min_waiting_result["breakdown"][2]
                    _append_case_metric(
                        detail_store["min_waiting"],
                        scenario_name,
                        cfg,
                        min_waiting_result["travel_cost_minutes"],
                        min_waiting_result["waiting_cost_minutes"],
                        min_waiting_result["route_failure_count"],
                        min_waiting_result["route_failures"],
                    )

                if using_shortest:
                    shortest_result = _run_evaluation_rollout(env, agent, cfg, route_planning_mode, rollout_mode = "shortest")
                    shortest_total_eval_reward += shortest_result["total_reward"]
                    shortest_total_traveling_time += shortest_result["breakdown"][0]
                    shortest_total_waiting_time += shortest_result["breakdown"][1]
                    shortest_total_charging_time += shortest_result["breakdown"][2]
                    _append_case_metric(
                        detail_store["shortest"],
                        scenario_name,
                        cfg,
                        shortest_result["travel_cost_minutes"],
                        shortest_result["waiting_cost_minutes"],
                        shortest_result["route_failure_count"],
                        shortest_result["route_failures"],
                    )

    result = (
        total_eval_reward,
        total_eval_min_waiting_reward,
        shortest_total_eval_reward,
        (total_traveling_time, total_waiting_time, total_charging_time),
        (min_waiting_total_traveling_time, min_waiting_total_waiting_time, min_waiting_total_charging_time),
        (shortest_total_traveling_time, shortest_total_waiting_time, shortest_total_charging_time),
    )
    if not return_details:
        return result

    detail = {
        "metric_definition": {
            "travel_cost_minutes": "Positive travel-time cost in minutes per testing case.",
            "waiting_cost_minutes": "Positive waiting-time cost in minutes per testing case.",
            "travel_waiting_total_minutes": "Positive total cost in minutes per testing case, computed as travel + waiting and excluding charging time.",
            "route_failure_count": "Number of EVs that did not finish under the route policy. Each receives a 50-minute censored travel cost.",
        },
        "evaluation_scope": {
            "probability_indices": [4, 6],
            "scenario_indices": flow_indices,
            "scenario_names": [_resolve_veh_flow(veh_flows, rd3) for rd3 in flow_indices],
            "seed_range": [1, 10],
            "general_data_dir": "higher_test_data",
            "power_data_dir": "higher_test_data_power",
        },
        "methods": {
            "agent": _summarize_method_metrics(detail_store["agent"]),
            "min_waiting": _summarize_method_metrics(detail_store["min_waiting"]) if using_min_waiting else None,
            "shortest": _summarize_method_metrics(detail_store["shortest"]) if using_shortest else None,
        },
    }
    return (*result, detail)
