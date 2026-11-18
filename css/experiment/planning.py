from __future__ import annotations

import torch
import traci
from traci.exceptions import TraCIException

from css.experiment.config import DECISION_DIST_THRESHOLD, MAX_DECISIONS_PER_EV
from css.graph import build_data_runtime, dynamic_node_feature_update
from css.sumo import (
    action_mask_dqn,
    apply_full_route,
    apply_next_junction,
    candidate_next_junctions,
    find_sumo_route_to_node,
    find_tls_and_link_index_for_movement,
    remaining_routes_len,
    time_to_green_and_remaining_green,
)
from css.sumo.traffic_signals import lane2edge, get_tls_ids, get_controlled_links

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


def oneHot(i, n, device="cpu"):
    if n <= 0:
        raise ValueError(f"one-hot size must be positive, got {n}")
    if not 0 <= i < n:
        raise ValueError(f"one-hot index {i} is outside [0, {n})")
    v = torch.zeros(n, device=device)
    v[i] = 1.0
    return v


def resolve_movement_out_lanes(cur_edge: str, out_edge_id: str) -> list[str]:
    lane_ids = []
    for tls in get_tls_ids(traci_api=traci):
        links = get_controlled_links(tls, traci_api=traci)
        for link_group in links:
            for in_lane, out_lane, _ in link_group:
                if lane2edge(in_lane) == cur_edge and lane2edge(out_lane) == out_edge_id:
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



def _prepare_route_graph(env, st, cur_edge, outgoing_edges, signal_state,
                         source, target, travel_time, mean_speed, occupancy):
    """Collect graph observations only for route policies that consume them."""
    requires_graph = (getattr(env.route_planning_agent, "requires_route_graph", True)
                      or getattr(env, "charging_uses_graph_state", False))
    if requires_graph:
        graph = build_data_runtime(
            env.sumo_static_pack, travel_time, mean_speed, occupancy,
            edge_feature_mode=getattr(env.route_planning_agent, "edge_feature_mode", None) or "all",
        )
    else:
        # Preserve visit-history bookkeeping without fetching global edge data.
        graph = env.sumo_static_pack["data_static"]
    st.recent_nodes.append(source)
    graph, st.ever_visit, st.visit_age = dynamic_node_feature_update(
        graph, target, st.recent_nodes, st.ever_visit, st.visit_age,
    )
    if requires_graph:
        graph.action_features = build_candidate_action_features(
            env, graph, st, cur_edge, outgoing_edges, st.neighbors, signal_state,
        )
        st.data_t = graph
    else:
        st.data_t = None


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
    if hasattr(env.route_planning_agent, "build_shared_observation"):
        state = env.route_planning_agent.build_shared_observation(state, tmp_traffic_light_state, traci.simulation.getTime())
    elif hasattr(env.route_planning_agent, "concat_state_time"):
        state = env.route_planning_agent.concat_state_time(
            state,
            traci.simulation.getTime(),
        )
    else:
        state = env.route_planning_agent.concat_state_tls(state, tmp_traffic_light_state)
    
    """
    6. 準備給GNN使用的graph feature
    """
    source = env.nid2idx[to_node]
    target = env.nid2idx[st.goal_cs]
    _prepare_route_graph(
        env, st, cur_edge, b_sorted, tmp_traffic_light_state,
        source, target, travel_time, MeanSpeed, Occupancy,
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
    if hasattr(env.route_planning_agent, "build_shared_observation"):
        state = env.route_planning_agent.build_shared_observation(state, tmp_traffic_light_state, traci.simulation.getTime())
    elif hasattr(env.route_planning_agent, "concat_state_time"):
        state = env.route_planning_agent.concat_state_time(
            state,
            traci.simulation.getTime(),
        )
    else:
        state = env.route_planning_agent.concat_state_tls(state, tmp_traffic_light_state)

    """
    6. 準備給GNN使用的graph feature
    """
    source = env.nid2idx[to_node]
    target = env.nid2idx[st.goal_cs]
    _prepare_route_graph(
        env, st, cur_edge, b_sorted, tmp_traffic_light_state,
        source, target, travel_time, MeanSpeed, Occupancy,
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
