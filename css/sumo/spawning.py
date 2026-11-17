from __future__ import annotations

import random
import time

import traci

from css.routing.algorithms import cost_by_traveltime, dijkstra_path
from css.sumo.route_files import save_routes_xml


def spawn_for_rl(veh_id: str, from_edge: str, vtype: str = "ev",
        depart_lane: str = "best", depart_pos: str = "base", depart_speed: str = "max"):
    route_id = f"tmp_{veh_id}"
    traci.route.add(route_id, [from_edge])
    traci.vehicle.add(veh_id, route_id, typeID=vtype,
                      departLane=depart_lane, departPos=depart_pos, departSpeed=depart_speed)


def spawn_evs_testing(base_depart=0, depart_gap=5.0, net = None, adj = None, src_edges = None, goal_nodes = None):
    """
    start_edges: 候選起點 external edges（字串 list）
    N: 要生幾台
    base_depart: 第一台的 depart 時間（秒）
    depart_gap: 每台出發間隔
    確定起點的edge跟終點的node即可
    用來判斷讀取.rou檔案跟在python中生成是否有差別
    """

    for i in range(len(src_edges)):
        ev = f"EV{i+1}"
        s_edge = src_edges[ev]
        cs = goal_nodes[ev]
        print(f"{ev}, s_edge {s_edge}, cs {cs}")
        rid = f"route_{ev}"
        traci.route.add(rid, [s_edge])
        traci.vehicle.add(
            vehID=ev, routeID=rid, depart=str(base_depart + i*depart_gap),
            departLane="best", departSpeed="max"
        )
        # 建議：速度/碰撞模式較寬鬆，避免奇怪報錯中斷
        traci.vehicle.setSpeedMode(ev, 32)


def spawn_evs(start_edges, cs_loc, N, base_depart=0, depart_gap=5.0, net = None, adj = None, filename = "None", using_env = " "):
    """
    start_edges: 候選起點 external edges（字串 list）
    N: 要生幾台
    base_depart: 第一台的 depart 時間（秒）
    depart_gap: 每台出發間隔
    """

    
    goal_nodes = {}
    ev_routes = []
    goal_node_list = random.choices(cs_loc, k = N)
    # goal_node_list[0] = cs_loc[0] # 在small_env測試同一個終點
    # with open(f"./{using_env}/goal_nodes/{filename}_goal_nodes.json", "w", encoding="utf-8") as f:
    #     json.dump(goal_node_list, f, ensure_ascii=False, indent=2)
        # a_loaded = json.load(f)
    
    waypoints = {}
    for i in range(N):
        ev = f"EV_{i+1}"
        # s_edge = start_edges[i % len(start_edges)]  # 或自行抽樣
        while True:
            s_edge = random.choice(start_edges)
            # print(f"s_edge {s_edge}")
            # cs = random.choice(cs_loc)
            cs = goal_node_list[i]
            tmp = dijkstra_path(net = net, adj = adj, cur_edge = s_edge, goal_node=cs, cost_fn=cost_by_traveltime)
            if not tmp[0] or len(tmp[1]) <= 2: # 不可抵達 or 起點剛好是終點
                continue
            # if len(tmp[1]) > 10: # 限制起點與終點距離不要太遠
            #     continue
            else: # 能抵達，計算waypoint
                path_nodes = tmp[0]
                path_edges = tmp[1]
                # print(f"path_edges {path_edges}")
                # sys.exit()
                tmp_i = 4
                step_i = 4
                tmp_waypoints = []
                while tmp_i <= len(path_nodes):
                    tmp_waypoints.append(path_nodes[tmp_i-1])
                    tmp_i += step_i
                if len(tmp_waypoints) == 0 or path_nodes[-1] != tmp_waypoints[-1]:
                    tmp_waypoints.append(path_nodes[-1])
                waypoints[ev] = []+tmp_waypoints
                break
        ev_routes.append([s_edge])
        goal_nodes[ev] = cs
        run_tag = str(int(time.time()*1000))
        ev  = f"EV{i+1}_{run_tag}"
        rid = f"route_{ev}"
        
        traci.route.add(rid, [s_edge])
        traci.vehicle.add(
            vehID=ev, routeID=rid, depart=str(base_depart + i*depart_gap),
            departLane="best", departSpeed="max"
        )
        # 建議：速度/碰撞模式較寬鬆，避免奇怪報錯中斷
        traci.vehicle.setSpeedMode(ev, 32)  # 權重可依需求

    save_routes_xml(ev_routes, filename, using_env = using_env)
    return goal_nodes, waypoints

