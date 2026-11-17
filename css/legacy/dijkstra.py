from __future__ import annotations

import json
from collections import deque

import torch
import traci

from css.experiment.departures import compress_route_on_depart, is_ev
from css.routing.algorithms import cost_by_traveltime, dijkstra_path
from css.sumo.distance import remaining_routes_len
from css.sumo.route_files import count_evs_in_rou
from css.sumo.routing import apply_full_route
from css.sumo.traffic_signals import reset_signal_topology_cache


def dijkstra_sim(net, static_pack, ev_ids, filesource, rd1 = None, rd2 = None, rd3 = None, rd4 = None, rd5 = None, test = False, set_goal = None, testing_paths = None,
                 traffic_jam = False, using_env = None, ev_route = None):
    class EVState:
        def __init__(self, n_nodes):
            self.done = False
            self.out = False #超出步數 or 超過範圍 
            self.decisions = 0
            self.total_cost = 0.0

            self.last_decision_node = None
            self.prev_decision_time = None
            self.prev_edge = None

            self.data_t = None
            self.recent_nodes = deque(maxlen=4)
            self.ever_visit = torch.zeros(n_nodes, 1)
            self.visit_age = torch.zeros(n_nodes, 1)

            self.internal_edges = None
            self.exit_lane_id = None

            self.last_committed_exit_edge = None
            self.current_exit_edge = None


            self.last_action_nid = None
            self.last_est_time = None

            self.neighbors = None
            self.neighbor_edges = None

            self.traffic_light_state = None
            self.routes = []

            self.time_spend = 0
    
    if traffic_jam:
        # INC_EDGE = "5707608#2"                                        ### traffic jam
        INC_EDGES = ["5707608#2" , "5707608#0"]                         ### traffic jam
        FACTOR   = 0.3          # 0.2~0.4 之間                           ### traffic jam
        t0, t1   = 120, 600     # 先用固定；也可以改成動態                  ### traffic jam

        ## get lane num
        lane_ids = []                                                   ### traffic jam
        base_lane_speed = {}                                            ### traffic jam
        for e in INC_EDGES:                                             ### traffic jam
            nlanes = traci.edge.getLaneNumber(e)                        ### traffic jam
            for i in range(nlanes):                                     ### traffic jam
                lid = f"{e}_{i}"                                        ### traffic jam
                lane_ids.append(lid)                                    ### traffic jam
                base_lane_speed[lid] = traci.lane.getMaxSpeed(lid)      ### traffic jam

        applied = False                                                 ### traffic jam
        reverted = False                                                ### traffic jam


    cfg_path = str(filesource / "osm.sumocfg")    # 要用新的 osm.sumocfg
    # cfg_path = str(filesource / "new_env.sumocfg")

    edge_ids = static_pack["edge_ids"]
    nid2idx = static_pack["nid2idx"]
    eid2eidx = static_pack['eid2eidx']
    adj = static_pack["adj"]
    decision_dist_threshold = 100.0
    goal_arrive_dist = 1.0
    def dist_to_lane_end(ev):
        lane_id = traci.vehicle.getLaneID(ev)
        pos = traci.vehicle.getLanePosition(ev)      # 從 lane 起點量到車頭的位置 (m)
        lane_len = traci.lane.getLength(lane_id)
        return max(0.0, lane_len - pos)

   
    ### 最新設定
    general_veh_flows = ["free_0900", "norm_1500", "busy_2100", "heavy_2400"]
    general = str(filesource / f"testing_data/output_{general_veh_flows[rd5]}_{rd3}.rou.xml") 
    ev_routes = str(f"./{using_env}/testing_data/ev_{general_veh_flows[rd5]}_p0{rd4}_{rd3}.rou.xml")
    if ev_route:
        ev_routes = ev_route
    # ev_routes = str(f"./{using_env}/new_training_data_3/{rd3}_training_set.rou.xml") ## 先用另外的測試

    evs = []
    evNum = count_evs_in_rou(ev_routes)
    for i in range(0,evNum): #small env改成5
        evs.append(f"EV_{i}")


    with open(f"./{using_env}/testing_data_goal/goal_ev_{general_veh_flows[rd5]}_p0{rd4}_{rd3}.json", "r", encoding="utf-8") as f:
        test_goal_node = json.load(f)
        test_goal_node = test_goal_node * 9

    goal_nodes = {}
    for idx,ev in enumerate(evs):
        goal_nodes[ev] = test_goal_node[idx]


    # print(general)
    # print(ev_routes)
    load_args = ["-c", cfg_path, "--step-length", "0.5", "--route-files", f"{general}, {ev_routes}","--no-warnings", "--tls.all-off"] #不使用紅綠燈
    load_args = ["-c", cfg_path, "--step-length", "0.5", "--route-files", f"{general}, {ev_routes}","--no-warnings"]
    
    reset_signal_topology_cache()
    traci.load(load_args)

    states = {ev: EVState(n_nodes=len(nid2idx)) for ev in evs}
    start_time = {ev: None for ev in evs}
    decided = {ev: False for ev in evs}
    def is_ev(vid: str) -> bool:
        EV_PREFIX = "EV_"
        return vid.startswith(EV_PREFIX)

    def compress_route_on_depart():
        """把剛出發的 EV route 壓成 [currentEdge]"""
        for vid in traci.simulation.getDepartedIDList():
            if not is_ev(vid):
                continue

            cur = traci.vehicle.getRoadID(vid)
            # 有時剛出發會在 internal edge 或拿到空值，保守處理
            if not cur or cur.startswith(":"):
                continue

            traci.vehicle.setRoute(vid, [cur])


    while True:
        traci.simulationStep()
        compress_route_on_depart()
        remaining_ev = traci.simulation.getMinExpectedNumber()
        if remaining_ev <= 0:
            break
        now = traci.simulation.getTime()
        # print(f"time {now}")
        if traffic_jam:
            if (not applied) and (now >= t0):                                   ### traffic jam
                for lid in lane_ids:                                                            ### traffic jam
                    traci.lane.setMaxSpeed(lid, base_lane_speed[lid] * FACTOR)                  ### traffic jam
                applied = True                                                                  ### traffic jam

        present = set(traci.vehicle.getIDList())
        if all(st.done for ev, st in states.items()):
            break
        for ev, st in states.items():
            if st.done or ev not in present or decided[ev]:
                continue

            remaining = remaining_routes_len(net, ev)
            if remaining >= decision_dist_threshold:
                continue
            # cur_edge = traci.vehicle.getRoadID(ev)
            route = traci.vehicle.getRoute(ev)
            cur_edge = traci.vehicle.getRoadID(ev)
            out_edge = route[-1]

            to_node = net.getEdge(cur_edge).getToNode().getID()
            if to_node == goal_nodes[ev]:
                if dist_to_lane_end(ev) > goal_arrive_dist:
                    # 還沒真的到終點，只是進到 decision 範圍 -> 不要結算、不要選新動作
                    continue
                st.done = True
                try:
                    st.time_spend = (traci.simulation.getTime() - start_time[ev])/60
                except Exception as exc:
                    raise RuntimeError(
                        f"Unable to finalize travel time for EV {ev!r}; "
                        f"start={start_time.get(ev)!r}, goal={goal_nodes.get(ev)!r}"
                    ) from exc
                continue

            path_nodes, path_edges, total = dijkstra_path(net = net, adj = adj, cur_edge = cur_edge, goal_node=goal_nodes[ev], cost_fn=cost_by_traveltime)
            st.routes = path_edges
            start_time[ev] = traci.simulation.getTime()
            # print(f"{ev}, dijkstra using path {path_edges}")
            apply_full_route(ev, cur_edge, path_edges, suffix_keep=None)
            decided[ev] = True

        arrived = set(traci.simulation.getArrivedIDList())  # 這一步剛到達終點的車
        for ev in arrived:
            if ev not in states:
                continue
            st = states[ev]
            if st.done or st.out:
                continue
            # ---- 結算最後一段 ----
            st.done = True
            step_time = (now - start_time[ev]) / 60.0  # 分鐘

            st.time_spend += step_time

    time_spend = 0
    for ev, st in states.items():
        time_spend += st.time_spend
        testing_paths.append(st.routes)

    if traffic_jam:
        for lid in lane_ids:                                        ### traffic jam
            traci.lane.setMaxSpeed(lid, base_lane_speed[lid])       ### traffic jam

    return time_spend
