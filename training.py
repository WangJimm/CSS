# from .graph_utils import *
import json
import os
import sys
import copy
import random
import time
import numpy as np
from tqdm import tqdm
from collections import deque
from SAC.sacd import *
from torch_geometric.loader import DataLoader
from config import (
    RUN_DIR,
    training_veh_flows,
    validation_veh_flows,
    testing_veh_flows,
    using_env,
    cfg_path,
    EVState,
    MAX_DECISIONS_PER_EV,
    DECISION_DIST_THRESHOLD,
    EMA_feature,
    ROUTE_PLANNING_MODE,
)
from traci.exceptions import FatalTraCIError, TraCIException
from utils_sumo import *
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path

@dataclass
class SumoLoadConfig:
    rd1: int
    rd2: int
    rd3: int
    veh_flow: str | int
    general: str
    ev_routes: str
    evs: List[str]
    load_args: List[str]
    # goal_nodes: Dict[str, str]   # 或 str，看你 goal node 型別
    ev_power : Dict[str, float]
    

@dataclass(frozen=True)
class ScenarioKey:
    scenario_index: int
    probability_index: int
    seed: int


def _default_flow_indices(veh_flows: List[str | int]) -> List[int]:
    return list(range(len(veh_flows)))


def _resolve_veh_flow(veh_flows: List[str | int], flow_index: int) -> str | int:
    try:
        return veh_flows[flow_index]
    except IndexError as exc:
        raise ValueError(
            f"flow_index {flow_index} is out of range for veh_flows={veh_flows}"
        ) from exc


def choose_training_scenario_key(
    rng: random.Random | None = None,
    veh_flows: List[str | int] | None = None,
    flow_indices: List[int] | None = None,
    probability_indices: List[int] | None = None,
    seed_start: int = 100,
    seed_end: int = 150,
):
    rng = rng or random.Random()
    veh_flows = veh_flows or training_veh_flows
    flow_indices = flow_indices or _default_flow_indices(veh_flows)
    probability_indices = probability_indices or [1, 2]
    return ScenarioKey(
        scenario_index=rng.choice(flow_indices),
        probability_index=rng.choice(probability_indices),
        seed=rng.randint(seed_start, seed_end),
    )


def build_training_scenario_schedule(
    num_episodes: int,
    schedule_seed: int = 42,
    veh_flows: List[str | int] | None = None,
    flow_indices: List[int] | None = None,
    probability_indices: List[int] | None = None,
    seed_start: int = 100,
    seed_end: int = 150,
):
    rng = random.Random(schedule_seed)
    return [
        choose_training_scenario_key(
            rng=rng,
            veh_flows=veh_flows,
            flow_indices=flow_indices,
            probability_indices=probability_indices,
            seed_start=seed_start,
            seed_end=seed_end,
        )
        for _ in range(num_episodes)
    ]


def _results_output_dir():
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_training_results(payload, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(payload), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)


def load_sumo(
    eval = False,
    rd1 = None,
    rd2 = None,
    rd3 = None,
    scenario_key: ScenarioKey | None = None,
    veh_flows: List[str | int] | None = None,
):
    if not eval:
        veh_flows = veh_flows or training_veh_flows
        ## training設定
        rd1 = random.randint(100,150) # seed
        rd2 = random.randint(1,2) # p
        rd3 = random.randint(0,len(veh_flows) - 1) # scenario

        # ## using toy env
        # rd1 = 100
        # rd2 = 2
        # rd3 = 3
        if scenario_key is not None:
            rd1 = scenario_key.seed
            rd2 = scenario_key.probability_index
            rd3 = scenario_key.scenario_index

        veh_flow = _resolve_veh_flow(veh_flows, rd3)
        general = str(RUN_DIR / f"training_data/output_{veh_flow}_{rd1}.rou.xml") 
        ev_routes = str(f"{RUN_DIR}/training_data/ev_{veh_flow}_p0{rd2}_{rd1}.rou.xml") 
        ev_power_name = str(f"{RUN_DIR}/training_data_power/power_ev_{veh_flow}_p0{rd2}_{rd1}.json") 
        

        evs = []
        evNum = count_evs_in_rou(ev_routes)
        for i in range(0,evNum): #small env改成5
            evs.append(f"EV_{i}")
        evs = get_evs_in_rou(ev_routes)
        evs = get_evs_in_rou(ev_routes)
        load_args = ["-c", cfg_path, "--step-length", "0.5", "--route-files", f"{general}, {ev_routes}","--no-warnings"]

        evs = get_evs_in_rou(ev_routes)
        with open(ev_power_name, "r", encoding="utf-8") as f:
            power = json.load(f)
        ev_powers = {}
        for idx,ev in enumerate(evs):
            ev_powers[ev] = power[idx]
    if eval:
        """
        rd1, rd2, rd3在外面決定好了 直接讀取對應的data
        """
        veh_flows = veh_flows or testing_veh_flows
        veh_flow = _resolve_veh_flow(veh_flows, rd3)
        general = str(RUN_DIR / f"testing_data/output_{veh_flow}_{rd1}.rou.xml") 
        ev_routes = str(f"{RUN_DIR}/testing_data/ev_{veh_flow}_p0{rd2}_{rd1}.rou.xml")
        ev_power_name = str(f"{RUN_DIR}/testing_data_power/power_ev_{veh_flow}_p0{rd2}_{rd1}.json") 
        evs = get_evs_in_rou(ev_routes)

        evs = []
        evNum = count_evs_in_rou(ev_routes)
        for i in range(0,evNum): #small env改成5
            evs.append(f"EV_{i}")

        load_args = ["-c", cfg_path, "--step-length", "0.5", "--route-files", f"{general}, {ev_routes}","--no-warnings"]

        with open(ev_power_name, "r", encoding="utf-8") as f:
            power = json.load(f)
        ev_powers = {}
        for idx,ev in enumerate(evs):
            ev_powers[ev] = power[idx]
        
    evs = get_evs_in_rou(ev_routes)
    ev_powers = {ev: power[idx] for idx, ev in enumerate(evs)}

    return SumoLoadConfig(rd1, rd2, rd3, veh_flow, general, ev_routes, evs, load_args, ev_powers)


def load_higher_eval_sumo(rd1, rd2, rd3, veh_flows: List[str | int] | None = None):
    veh_flows = veh_flows or testing_veh_flows
    veh_flow = _resolve_veh_flow(veh_flows, rd3)
    general = str(RUN_DIR / f"higher_test_data/output_{veh_flow}_{rd1}.rou.xml")
    ev_routes = str(f"{RUN_DIR}/higher_test_data/ev_{veh_flow}_p0{rd2}_{rd1}.rou.xml")
    ev_power_name = str(f"{RUN_DIR}/higher_test_data_power/power_ev_{veh_flow}_p0{rd2}_{rd1}.json")

    evs = []
    evNum = count_evs_in_rou(ev_routes)
    for i in range(0, evNum):
        evs.append(f"EV_{i}")
    evs = get_evs_in_rou(ev_routes)

    load_args = ["-c", cfg_path, "--step-length", "0.5", "--route-files", f"{general}, {ev_routes}", "--no-warnings"]

    with open(ev_power_name, "r", encoding="utf-8") as f:
        power = json.load(f)
    ev_powers = {}
    for idx, ev in enumerate(evs):
        ev_powers[ev] = power[idx]

    return SumoLoadConfig(rd1, rd2, rd3, veh_flow, general, ev_routes, evs, load_args, ev_powers)

def check_arrived(states, env):
    """
    檢查這一步抵達的車當中是否有EV，有的話紀錄後續reward與state
    """
    arrived = set(traci.simulation.getArrivedIDList())  # 這一步剛到達終點的車
    for ev in arrived:
        if ev not in states:
            continue
        # ---- 結算最後一段 ----
        st = states[ev]
        st.done = True
        now = traci.simulation.getTime()
        st.travel_time = now - st.depart_time
        ### 這邊開始計算waiting time跟charging time
        waiting_time, charging_time = env.update(now, st.goal_cs, st.power)
        st.waiting_time = waiting_time
        st.charging_time = charging_time
        



def oneHot(i, n=27, device="cpu"):
    v = torch.zeros(n, device=device)
    v[i] = 1.0
    return v


def resolve_movement_out_lanes(cur_edge: str, out_edge_id: str) -> list[str]:
    lane_ids = []
    for tls in traci.trafficlight.getIDList():
        links = traci.trafficlight.getControlledLinks(tls)
        for link_group in links:
            for in_lane, out_lane, _ in link_group:
                if in_lane.startswith(cur_edge) and out_lane.startswith(out_edge_id):
                    lane_ids.append(out_lane)
    return list(dict.fromkeys(lane_ids))


def candidate_lane_metrics(cur_edge: str, edge):
    lane_ids = resolve_movement_out_lanes(cur_edge, edge.getID())
    if not lane_ids:
        lane_ids = [lane.getID() for lane in edge.getLanes()]

    halting_total = 0.0
    vehicle_total = 0.0
    for lane_id in lane_ids:
        try:
            halting_total += float(traci.lane.getLastStepHaltingNumber(lane_id))
        except Exception:
            pass
        try:
            vehicle_total += float(traci.lane.getLastStepVehicleNumber(lane_id))
        except Exception:
            pass

    return {
        "halting_norm": min(halting_total / 20.0, 1.0),
        "vehicle_norm": min(vehicle_total / 20.0, 1.0),
    }


def build_candidate_action_features(env, graph_data, st, cur_edge, outgoing_edges, neighbors, tmp_traffic_light_state):
    edge_attr_dim = int(graph_data.edge_attr.size(1))
    rows = []
    previous_node = st.recent_nodes[-2] if len(st.recent_nodes) >= 2 else None

    for idx, edge in enumerate(outgoing_edges):
        edge_id = edge.getID()
        edge_idx = env.eid2eidx.get(edge_id)
        if edge_idx is None:
            edge_attr = torch.zeros(edge_attr_dim, dtype=torch.float32)
        else:
            edge_attr = graph_data.edge_attr[edge_idx].detach().cpu().float()

        neighbor_idx = neighbors[idx]
        is_uturn = 1.0 if previous_node is not None and neighbor_idx == previous_node else 0.0
        visited_flag = float(st.ever_visit[neighbor_idx].item())
        visit_age = float(st.visit_age[neighbor_idx].item())
        tls_tensor = torch.tensor(tmp_traffic_light_state[idx], dtype=torch.float32)
        lane_metrics = candidate_lane_metrics(cur_edge, edge)
        extra = torch.tensor(
            [
                is_uturn,
                visited_flag,
                visit_age,
                float(lane_metrics["halting_norm"]),
                float(lane_metrics["vehicle_norm"]),
            ],
            dtype=torch.float32,
        )
        rows.append(torch.cat([edge_attr, tls_tensor, extra], dim=0))

    if not rows:
        return torch.zeros((0, edge_attr_dim + 7), dtype=torch.float32)
    return torch.stack(rows, dim=0)

def first_plan(ev, st, env, travel_time, MeanSpeed, Occupancy, evaluate = True):
    """
    1. 獲得EV現在位置
    """
    cur_edge = traci.vehicle.getRoadID(ev)
    to_node = env.net.getEdge(cur_edge).getToNode().getID()
    if to_node == st.goal_cs:
        return 

    """
    2. 從現在位置找到neighbors跟action_mask
    """
    jns, outs = candidate_next_junctions(env.net, cur_edge)
    st.neighbors = [env.nid2idx[n] for n in jns]
    action_mask = action_mask_dqn(env.net, cur_edge, env.nid2idx, n = env.route_planning_agent.num_nodes)

    """
    3. 用現在位置轉換成可以給DQN使用的狀態 state
    """ 
    s1 = oneHot(env.nid2idx[to_node], len(env.nid2idx))
    s2 = oneHot(env.nid2idx[st.goal_cs], len(env.nid2idx))
    state = torch.cat((s1,s2))

    """
    4. 根據neighbors去收集附近的紅綠燈狀態 tmp_traffic_light_state
    """
    # 找time_to_green remaining_green
    b_dict = {edge.getToNode().getID(): edge for edge in outs}
    b_sorted = [b_dict[to] for to in jns]
    tmp_traffic_light_state = []
    for e in b_sorted:
        tls, idx = find_tls_and_link_index_for_movement(cur_edge, e.getID())
        time_to_green, remaining_g_time = 0.0, 0.0
        if tls:
            time_to_green, remaining_g_time = time_to_green_and_remaining_green(tls, idx)
        tmp_traffic_light_state.append([time_to_green/60, remaining_g_time/60])

    """
    5. 把紅綠燈狀態(tmp_traffic_light_state) 跟位置狀態(state) 融合，變成完整的DQN state
    """
    ##融合state與tls state
    state = env.route_planning_agent.concat_state_tls(state, tmp_traffic_light_state) #len(N) + max_neihbor_num * 2
    
    """
    6. 準備給GNN使用的graph feature
    """
    source = env.nid2idx[to_node]
    target = env.nid2idx[st.goal_cs]
    st.data_t = build_data_runtime(
        env.sumo_static_pack,
        travel_time,
        MeanSpeed,
        Occupancy,
        edge_feature_mode=getattr(env.route_planning_agent, "edge_feature_mode", None) or "all",
    )
    st.recent_nodes.append(env.nid2idx[to_node])
    st.data_t, st.ever_visit, st.visit_age = dynamic_node_feature_update(st.data_t, env.nid2idx[st.goal_cs],
                                                                    st.recent_nodes, st.ever_visit, st.visit_age)
    st.data_t.action_features = build_candidate_action_features(
        env,
        st.data_t,
        st,
        cur_edge,
        b_sorted,
        st.neighbors,
        tmp_traffic_light_state,
    )

    
    """
    7. 把前面準備的state一起丟給DQNGNN做選擇
    """
    st.last_action_nid = env.route_planning_agent.choose_action_gnn(state, action_mask, st.neighbors, source, target, st.data_t, evaluate = evaluate)

    """
    8. 紀錄狀態
    """
    st.idx = st.neighbors.index(st.last_action_nid)
    next_jn = jns[st.idx]
    chosen_edge = apply_next_junction(ev, next_jn, outs)
    st.routes.append(chosen_edge)
    st.last_committed_exit_edge = st.current_exit_edge
    st.current_exit_edge = chosen_edge
    st.decision += 1


def second_plan(ev, st, env, travel_time, MeanSpeed, Occupancy, evaluate = True):
    """
    1. 獲得EV現在位置
    """
    cur_edge = st.current_exit_edge
    to_node = env.net.getEdge(cur_edge).getToNode().getID()

    """
    2. 從現在位置找到neighbors跟action_mask
    """
    jns, outs = candidate_next_junctions(env.net, cur_edge)
    st.neighbors = [env.nid2idx[n] for n in jns]
    action_mask = action_mask_dqn(env.net, cur_edge, env.nid2idx, n = env.route_planning_agent.num_nodes)

    """
    3. 用現在位置轉換成可以給DQN使用的狀態 state
    """ 
    s1 = oneHot(env.nid2idx[to_node], len(env.nid2idx))
    s2 = oneHot(env.nid2idx[st.goal_cs], len(env.nid2idx))
    state = torch.cat((s1,s2))

    """
    4. 根據neighbors去收集附近的紅綠燈狀態 tmp_traffic_light_state
    """
    # 找time_to_green remaining_green
    b_dict = {edge.getToNode().getID(): edge for edge in outs}
    b_sorted = [b_dict[to] for to in jns]
    tmp_traffic_light_state = []
    for e in b_sorted:
        tls, idx = find_tls_and_link_index_for_movement(cur_edge, e.getID())
        time_to_green, remaining_g_time = 0.0, 0.0
        if tls:
            time_to_green, remaining_g_time = time_to_green_and_remaining_green(tls, idx)
        tmp_traffic_light_state.append([time_to_green/60, remaining_g_time/60])

    """
    5. 把紅綠燈狀態(tmp_traffic_light_state) 跟位置狀態(state) 融合，變成完整的DQN state
    """
    ##融合state與tls state
    state = env.route_planning_agent.concat_state_tls(state, tmp_traffic_light_state) #len(N) + max_neihbor_num * 2

    """
    6. 準備給GNN使用的graph feature
    """
    source = env.nid2idx[to_node]
    target = env.nid2idx[st.goal_cs]
    st.data_t = build_data_runtime(
        env.sumo_static_pack,
        travel_time,
        MeanSpeed,
        Occupancy,
        edge_feature_mode=getattr(env.route_planning_agent, "edge_feature_mode", None) or "all",
    )
    st.recent_nodes.append(env.nid2idx[to_node])
    st.data_t, st.ever_visit, st.visit_age = dynamic_node_feature_update(st.data_t, env.nid2idx[st.goal_cs],
                                                                    st.recent_nodes, st.ever_visit, st.visit_age)
    st.data_t.action_features = build_candidate_action_features(
        env,
        st.data_t,
        st,
        cur_edge,
        b_sorted,
        st.neighbors,
        tmp_traffic_light_state,
    )


    """
    different. 跟first plan不一樣的地方，這邊開始計算reward
    """
    # 終點 & neighbors為空集合狀況
    if to_node == st.goal_cs or not jns:
        return 
    """
    7. 把前面準備的state一起丟給DQNGNN做選擇
    """
    st.last_action_nid = env.route_planning_agent.choose_action_gnn(state, action_mask, st.neighbors, source, target, st.data_t, evaluate = evaluate)


    """
    8. 紀錄狀態
    """
    st.idx = st.neighbors.index(st.last_action_nid)
    next_jn = jns[st.idx]
    chosen_edge = apply_next_junction(ev, next_jn, outs)
    st.routes.append(chosen_edge)
    st.last_committed_exit_edge = st.current_exit_edge
    st.current_exit_edge = chosen_edge
    st.decision += 1
        
   

def route_planning(states, env, travel_time, MeanSpeed, Occupancy):
    """
    每次推一個step，讓route planning agent依序決定路線
    """
    present = set(traci.vehicle.getIDList())

    for ev, st in states.items():
        if st.done or st.out or ev not in present or st.decision >= MAX_DECISIONS_PER_EV:
            continue

        try:
            remaining = remaining_routes_len(env.net, ev)
        except:
            print("[Warning] remaining_routes_len error")
            break
        if remaining >= DECISION_DIST_THRESHOLD:
            continue


        if st.current_exit_edge is None:
            # print(traci.vehicle.getRoute(ev))
            first_plan(ev, st, env, travel_time, MeanSpeed, Occupancy)
            # print(traci.vehicle.getRoute(ev))
        elif (st.current_exit_edge != st.last_committed_exit_edge):
            second_plan(ev, st, env, travel_time, MeanSpeed, Occupancy)
            # print(traci.vehicle.getRoute(ev))

def route_planning_dijkstra(states, env, decided):
    """
    這個版本實現用dijkstra的方式
    """
    

    remaining_ev = traci.simulation.getMinExpectedNumber()
    if remaining_ev <= 0:
        return
    present = set(traci.vehicle.getIDList())
    if all(st.done for ev, st in states.items()):
        return
    for ev, st in states.items():
        if st.done or ev not in present or decided[ev] or not st.goal_cs:
            continue

        current_road_id = traci.vehicle.getRoadID(ev)
        # SUMO has already committed the vehicle to a specific movement while
        # it is on a junction-internal edge. Replacing the route at that moment
        # may select a different exit and is rejected as "leading elsewhere".
        # Keep decided=False and retry on the next step, after the EV reaches
        # the next external edge.
        if not current_road_id or current_road_id.startswith((':', '!')):
            continue

        # Dijkstra/SUMO routing plans the complete trip in one shot. Unlike the
        # junction-by-junction DQNGNN planner, it must not wait until the old
        # route has less than DECISION_DIST_THRESHOLD remaining: that can carry
        # the EV onto the old route's terminal edge, from which the selected
        # charging station may no longer be reachable.
        route = traci.vehicle.getRoute(ev)
        route_index = traci.vehicle.getRouteIndex(ev)
        if route_index < 0 or route_index >= len(route):
            raise RuntimeError(
                f'Cannot reroute {ev}: route_index={route_index}, route={route}, '
                f'road_id={current_road_id}'
            )
        # getRoadID may be an internal edge such as :junction_0. SUMO routes do
        # not contain internal edges; route[getRouteIndex()] is the external
        # edge SUMO considers current for route replacement.
        route_anchor = route[route_index]
        if route_anchor != current_road_id:
            # This can occur in a transient route-index state. Do not submit a
            # route that does not start with the external edge occupied now.
            continue
        
        to_node = env.net.getEdge(route_anchor).getToNode().getID()
        if to_node == st.goal_cs:
            decided[ev] = True
            continue

        try:
            cached_route = getattr(st, 'planned_sumo_route', None)
            if cached_route and cached_route[0] == route_anchor:
                planned_route = list(cached_route)
            else:
                planned_route = find_sumo_route_to_node(
                    ev=ev,
                    net=env.net,
                    route_anchor=route_anchor,
                    goal_node=st.goal_cs,
                )
            # planned_route already includes route_anchor. Keep st.routes in
            # its original format (only the suffix after the current edge).
            st.routes = planned_route[1:]
            apply_full_route(
                ev,
                route_anchor,
                st.routes,
                suffix_keep=None,
                net=env.net,
            )
        except (TraCIException, ValueError) as exc:
            raise RuntimeError(
                f'Route replacement failed for {ev}: road_id={current_road_id}, '
                f'route_index={route_index}, anchor={route_anchor}, old_route={route}, '
                f'goal={st.goal_cs}. Error: {exc}'
            ) from exc
        decided[ev] = True
            
        

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
    seed_start: int = 1, # int = 100,  for env:1 / for small env: 100
    seed_end: int = 20, # int = 150, for env:20 / for small env: 150
    save_results: bool = True,
    results_filename: str | None = None,
    show_sumo_warnings: bool = False,
    checkpoint_filename: str | Path | None = None,
):
    """
    using_min_waiting: 若True, 會增加 min waiting time baseline 的模擬

    """
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
        veh_flows=training_veh_flows,
        flow_indices=flow_indices,
        probability_indices=probability_indices,
        seed_start=seed_start,
        seed_end=seed_end,
    )
    episode_records = []
    evaluation_records = []
    best_data = None
    base_data_1 = None
    base_data_2 = None
    best_evaluation_detail = None
    training_buffer = agent.memory if hasattr(agent, "memory") else replay_buffer
    required_buffer_size = max(1, int(getattr(agent, "batch_size", 1)))
    sumo_error_log = _results_output_dir() / 'sumo_training_errors.log'
    for epi_i in tqdm(range(episodes)):

        episode_reward = 0
        min_waiting_reward = 0
        scenario_key = scenario_schedule[epi_i]
        cfg = load_sumo(eval = False, scenario_key = scenario_key) # cfg相關去看SumoLoadConfig
        load_args = list(cfg.load_args)
        if show_sumo_warnings:
            load_args = [arg for arg in load_args if arg != '--no-warnings']
        elif '--no-warnings' not in load_args:
            load_args.append('--no-warnings')
        load_args.extend(['--error-log', str(sumo_error_log)])
        try:
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
            if route_planning_mode == "dqngnn":
                env.route_planning_agent = env.route_planning_agent_dqngnn
                route_planning(states, env, travel_time, MeanSpeed, Occupancy)        ## 開始task 2, route planning
            elif route_planning_mode == "dqn":
                env.route_planning_agent = env.route_planning_agent_dqn
                route_planning(states, env, travel_time, MeanSpeed, Occupancy)        ## 開始task 2, route planning
            elif route_planning_mode == "edqn":
                env.route_planning_agent = env.route_planning_agent_edqn
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
        if incomplete_evs:
            examples = '; '.join(
                f'{ev}(goal={states[ev].goal_cs}, decisions={states[ev].decision}, '
                f'last_edges={states[ev].routes[-3:]})'
                for ev in incomplete_evs[:10]
            )
            raise RuntimeError(
                f'Training episode {epi_i} ended with {len(incomplete_evs)} unfinished EVs '
                f'(examples: {examples}); seed={cfg.rd1}, probability={cfg.rd2}, '
                f'flow={cfg.veh_flow}. No incomplete reward was added to the agent. '
                f'Check MAX_DECISIONS_PER_EV and the route-planning policy.'
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
                    veh_flows = validation_veh_flows,
                    flow_indices = validation_flow_indices,
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
                    veh_flows = validation_veh_flows,
                    flow_indices = validation_flow_indices,
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
        output_path = _results_output_dir() / results_filename
        payload = {
            "summary": {
                "episodes": episodes,
                "run_evaluation": run_evaluation,
                "using_min_waiting": using_min_waiting,
                "route_planning_mode": route_planning_mode,
                "training_start_episode": training_start_episode,
                "schedule_seed": schedule_seed,
                "training_veh_flows": training_veh_flows,
                "validation_veh_flows": validation_veh_flows,
                "training_flow_indices": flow_indices or _default_flow_indices(training_veh_flows),
                "validation_flow_indices": validation_flow_indices or _default_flow_indices(validation_veh_flows),
                "probability_indices": probability_indices or [1, 2],
                "seed_start": seed_start,
                "seed_end": seed_end,
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
    }


def _append_case_metric(store, scenario_name, cfg, travel_cost_minutes, waiting_cost_minutes):
    if scenario_name not in store:
        store[scenario_name] = _empty_case_metric_store()
    total_cost_minutes = travel_cost_minutes + waiting_cost_minutes
    store[scenario_name]["travel_cost_minutes"].append(float(travel_cost_minutes))
    store[scenario_name]["waiting_cost_minutes"].append(float(waiting_cost_minutes))
    store[scenario_name]["travel_waiting_total_minutes"].append(float(total_cost_minutes))
    store[scenario_name]["cases"].append(
        {
            "seed": int(cfg.rd1),
            "probability_index": int(cfg.rd2),
            "scenario_index": int(cfg.rd3),
            "scenario_name": scenario_name,
            "travel_cost_minutes": float(travel_cost_minutes),
            "waiting_cost_minutes": float(waiting_cost_minutes),
            "travel_waiting_total_minutes": float(total_cost_minutes),
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
            "cases": overall_cases,
        },
    }


def _run_evaluation_rollout(env, agent, cfg, route_planning_mode, rollout_mode = "agent"):
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

    while True:
        try:
            remaining_ev = traci.simulation.getMinExpectedNumber()
        except (FatalTraCIError, TraCIException, OSError):
            print("[Warning] SUMO closed during getMinExpectedNumber()")
            break
        if remaining_ev <= 0:
            break
        try:
            traci.simulationStep()
        except (FatalTraCIError, TraCIException, OSError):
            print("[Warning] SUMO closed during simulationStep()")
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

        if route_planning_mode == "dqngnn":
            env.route_planning_agent = env.route_planning_agent_dqngnn
            route_planning(states, env, travel_time, MeanSpeed, Occupancy)
        elif route_planning_mode == "dqn":
            env.route_planning_agent = env.route_planning_agent_dqn
            route_planning(states, env, travel_time, MeanSpeed, Occupancy)
        elif route_planning_mode == "edqn":
            env.route_planning_agent = env.route_planning_agent_edqn
            route_planning(states, env, travel_time, MeanSpeed, Occupancy)
        elif route_planning_mode == "dijkstra":
            route_planning_dijkstra(states, env, decided)

        check_arrived(states, env)
        if all(st.done or st.out or st.decision >= MAX_DECISIONS_PER_EV for ev, st in states.items()):
            break

    total_reward = 0.0
    total_traveling_time = 0.0
    total_waiting_time = 0.0
    total_charging_time = 0.0
    travel_cost_minutes = 0.0
    waiting_cost_minutes = 0.0
    for ev, st in states.items():
        if st.travel_time is None:
            print(
                "[INCOMPLETE EV]",
                ev,
                "done=", st.done,
                "out=", st.out,
                "depart_time=", st.depart_time,
                "goal_cs=", st.goal_cs,
                "decision=", st.decision,
            )

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
):
    veh_flows = veh_flows or testing_veh_flows
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

    for rd1 in tqdm(range(51, 61)):
        for rd2 in range(1, 3):
            for rd3 in flow_indices:
                cfg = load_sumo(eval = True, rd1 = rd1, rd2 = rd2, rd3 = rd3, veh_flows = veh_flows)
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
        },
        "evaluation_scope": {
            "probability_indices": [1, 2],
            "scenario_indices": flow_indices,
            "scenario_names": [_resolve_veh_flow(veh_flows, rd3) for rd3 in flow_indices],
            "seed_range": [51, 60],
            "general_data_dir": "testing_data",
            "power_data_dir": "testing_data_power",
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
):
    veh_flows = veh_flows or testing_veh_flows
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
                cfg = load_higher_eval_sumo(rd1 = rd1, rd2 = rd2, rd3 = rd3, veh_flows = veh_flows)
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


