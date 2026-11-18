from __future__ import annotations

import random

import numpy as np
import traci

from css.sumo.routing import find_sumo_route_to_node


def get_dis_from_path(path, net):
    total_d = 0
    for p in path:
        e = net.getEdge(p)
        lane_id0 = e.getLanes()[0].getID()
        d = traci.lane.getLength(lane_id0)
        total_d += d
    return total_d


def is_ev(vid: str) -> bool:
    EV_PREFIX = "EV_"
    return vid.startswith(EV_PREFIX)


def compress_route_on_depart(
    states,
    env,
    agent,
    travel_time=None,
    MeanSpeed=None,
    Occupancy=None,
    eval=False,
    using_min_waiting=False,
    using_shortest=False,
    using_random=False,
    truncate_departure_route=True,
):
    """把剛出發的 EV route 壓成 [currentEdge], 並且做出cs選擇"""
    # Graph-based station selectors share these EMAs with route observations.
    # Retain their original update cadence even when the route DQN ignores graphs.
    env.charging_uses_graph_state = bool(getattr(agent, "uses_graph_state", False))
    for vid in traci.simulation.getDepartedIDList():
        """
        1. 把EV route 壓縮成[cur edge]
        """
        if not is_ev(vid):
            continue
        cur = traci.vehicle.getRoadID(vid)
        # 有時剛出發會在 internal edge 或拿到空值，保守處理
        if not cur or cur.startswith(":"):
            continue
        if truncate_departure_route:
            traci.vehicle.setRoute(vid, [cur])
        # route = traci.vehicle.getRoute(vid)
        # print(vid,route)

        """
        2. 紀錄出發時間
        """
        states[vid].depart_time = traci.simulation.getTime()

        """
        3. 選擇cs
        """
        if states[vid].goal_cs is not None:
            raise RuntimeError(
                f"EV {vid!r} already has charging-station goal "
                f"{states[vid].goal_cs!r} before departure selection"
            )
        to_node = env.net.getEdge(cur).getToNode().getID()
        
        """
        3.1 random (平均分配到各充電站，等待時間應該會比較少)
        """
        if using_min_waiting or using_random:
            states[vid].goal_cs = env.get_min_waiting_time_cs(states[vid].depart_time)
            continue

        """
        3.2 最短距離 (單純選擇距離最短的充電站，旅行時間比較少)
        """
        if using_shortest:
            states[vid].goal_cs = env.cs_loc[np.argmin(env.dis_table[to_node])]
            continue
        
        """
        3.3 dqn (從互動中學習)
        """
        ev_power = env.ev_powers[vid]
        if getattr(agent, 'uses_contextual_state', False):
            sacd_state = agent.build_state(states[vid].depart_time, to_node, ev_power, travel_time, MeanSpeed, Occupancy)
            feasible_routes = {}
            for candidate_action, candidate_cs in enumerate(env.cs_loc):
                try:
                    feasible_routes[candidate_action] = find_sumo_route_to_node(
                        ev=vid,
                        net=env.net,
                        route_anchor=cur,
                        goal_node=candidate_cs,
                    )
                except ValueError:
                    continue
            if not feasible_routes:
                raise RuntimeError(
                    f'No charging station is SUMO-reachable for {vid} '
                    f'from departure edge {cur}'
                )
            available_actions = sorted(feasible_routes)
            action = agent.choose_action(
                sacd_state,
                eval=eval,
                available_actions=available_actions,
            )
            states[vid].feasible_cs_actions = available_actions
            states[vid].planned_sumo_route = feasible_routes[action]
        elif hasattr(agent, "uses_graph_state") and agent.uses_graph_state:
            sacd_state = agent.build_state(states[vid].depart_time, to_node, ev_power, travel_time, MeanSpeed, Occupancy)
            action = agent.choose_action(sacd_state, eval=eval)
        else:
            sacd_state = env.generate_sacd_state(states[vid].depart_time, to_node, ev_power)
            action = agent.choose_action(sacd_state, eval)
        # print(vid, sacd_state, env.cs_loc[action])
        states[vid].goal_cs = env.cs_loc[action]
        states[vid].action = action
        states[vid].sacd_state = sacd_state
        if (not eval) and hasattr(agent, "EPSILON") and hasattr(agent, "min_epsilon"):
            agent.EPSILON = max(
                agent.min_epsilon,
                agent.EPSILON * 0.995
            )


def choose_action_random(cs_loc):
    """
    暫時先用random跑看看，確認架構沒問題就換成agnet的choose_action
    """
    return random.choice(cs_loc)
