"""
utils for sumo
"""
import traci
import sys
import numpy as np
import torch
from torch_geometric.data import Data
from traci import constants as tc
from collections import defaultdict
import heapq
from collections import deque
import random
import matplotlib.pyplot as plt
import json
import copy
import xml.etree.ElementTree as ET

import time



MIN_DECISION_LEN = 2.0       # 小於這長度視為「太短」、不可單獨決策
MAX_SKIP_HOPS    = 8         # 最多往前穿越幾條邊，避免爆炸展開
VCLASS_FILTER    = None         # 依你的 vType 調整或設為 None
DECAY = 0.95
FLOOR = 0


def is_external_node(n):
    return (n.getType() != "internal") and (not n.getID().startswith(":"))
def is_external_edge(e):
    f = e.getFunction()
    return (f != "internal") and (not e.getID().startswith(":"))

def save_memory_to_txt(buffer, filename="memory_buffer.txt"):
    """
    將 replay buffer 輸出成 .txt 檔案
    :param buffer: 例如 agent.memory.buffer (list / deque / array)
    :param filename: 儲存檔案名稱
    """
    import os

    # 預設存到當前資料夾
    filepath = os.path.join(os.getcwd(), filename)

    with open(filepath, "w", encoding="utf-8") as f:
        for i, item in enumerate(buffer):
            f.write(f"{i}: {str(item)}\n")

    print(f"[INFO] buffer 已儲存到 {filepath}")
    return filepath


def dynamic_node_feature_update(data_static, target, recent_node, ever_visit, visit_age):
    """
    更新data_static的動態node feature部分
    target: 用的是轉換後的編號
    recent_node: deque, 最近幾次拜訪過的點
    ever_visit: torch [node_len, 1], 曾經拜訪過的點 
    visit_age: torch [node_len, 1], 最近拜訪的點，越近拜訪分數越大
    """

    is_target = torch.zeros(len(data_static.x))
    is_target[target] = 1
    is_target = is_target.unsqueeze(1)

    ever_visit[recent_node[-1]] = 1

    recent_flag = torch.zeros(len(data_static.x), 1)
    for nid in recent_node:
        recent_flag[nid] = 1.0


    visit_age[:] = np.maximum(DECAY * visit_age, FLOOR)
    visit_age[recent_node[-1]] = 1.0

    new_x = torch.cat((data_static.x, is_target, ever_visit, recent_flag, visit_age), dim = 1)

    new_data_static = Data(
        x=new_x,
        edge_index=data_static.edge_index,
        edge_attr=data_static.edge_attr,
    )

    return new_data_static, ever_visit, visit_age


def get_edge_index_v2(net):
    """
    這邊使用junction當node
    node: (static)in/out_degree, is_signalized  (dynamic)is_target, recent_flag, visit_age
    edge: (static)length, speed_limit, num_lanes (dynamic)avg_speed, travel_time
    """
    # 節點（junction）
    nodes = [n for n in net.getNodes() if is_external_node(n)]
    nid2idx = {n.getID(): i for i, n in enumerate(nodes)}

    ext_edges = [e for e in net.getEdges() if is_external_edge(e)]


    # node features（示例：是否有號誌、入/出度）
    Xn = np.zeros((len(nodes), 3), np.float32)
    for n in nodes:
        i = nid2idx[n.getID()]
        tls = 1.0 if n.getType().startswith("traffic_light") else 0.0
        indeg, outdeg = len([e for e in n.getIncoming() if e.getFunction() != "internal"]), len([e for e in n.getOutgoing() if e.getFunction() != "internal"])
        Xn[i] = [tls, indeg, outdeg]

    
    # edge_index 與靜態 edge_attr（長度、限速、車道數）
    u, v, Xe = [], [], []
    for e in ext_edges:
        fr, to = e.getFromNode().getID(), e.getToNode().getID()
        if fr in nid2idx and to in nid2idx:
            u.append(nid2idx[fr]); v.append(nid2idx[to])
            Xe.append([e.getLength(), e.getSpeed(), len(e.getLanes())])

    edge_index = torch.tensor([u, v], dtype=torch.long)
    data_static = Data(
        x=torch.from_numpy(Xn),
        edge_index=edge_index,
        edge_attr=torch.tensor(np.array(Xe), dtype=torch.float32),
    )
    # 保留對應，之後動態取值要用
    edge_ids = [e.getID() for e in ext_edges]
    # print(edge_index)

    adj = defaultdict(list) # adj紀錄每個edge的長度
    edge_len = {}
    for e in ext_edges:
        # e = <edge id="A0A1" from="A0" to="A1"/>
        u = e.getFromNode().getID()
        v = e.getToNode().getID()
        # u = A0, v = A1
        eid = e.getID()
        # eid = A0A1
        edge_len[eid] = e.getLength() # 可以直接獲取該路段長度
        if (u in nid2idx) and (v in nid2idx):
            adj[u].append((v, eid))
    
    return data_static, edge_ids, nid2idx, adj, edge_len

def prepare_static(net):
    data_static, edge_ids, nid2idx, adj, edge_len = get_edge_index_v2(net)

    # 補: eid2eidx、靜態 edge 特徵（對齊 PyG 的邊順序）
    eid2eidx = {eid: i for i, eid in enumerate(edge_ids)}
    # 從 data_static.edge_attr 取出靜態欄位（長度、速限、車道數）
    static_edge_attr = data_static.edge_attr.clone()  # (E, 3) = [length, speed, lanes]

    # 建議先做簡單正規化（避免尺度差太大）
    # 這裡示範 min-max；你也可換 z-score（需保存均值/標準差）
    def minmax(t):
        mn, mx = t.min(), t.max()
        return (t - mn) / (mx - mn + 1e-8), (mn, mx)
    ## 這邊插入交通號誌特徵
    tls_pool = traci.trafficlight.getIDList()
    pyg_edge_in_edge_ids = edge_ids
    pyg_edge_target_tls_ids = []
    for e in pyg_edge_in_edge_ids:
        v = net.getEdge(e).getToNode().getID()
        if v in tls_pool:
            pyg_edge_target_tls_ids.append(v)
        else:
            pyg_edge_target_tls_ids.append(None)
    cycle_len, green_rat = compute_signal_backbone_for_edges(pyg_edge_in_edge_ids, pyg_edge_target_tls_ids)
    cycle_len = torch.tensor(cycle_len, dtype=torch.float32)
    cycle_len /= 60
    green_rat = torch.tensor(green_rat, dtype=torch.float32)
    length_n, len_mm = minmax(static_edge_attr[:, 0])
    speed_n, spd_mm = minmax(static_edge_attr[:, 1])
    lanes_n, lns_mm = minmax(static_edge_attr[:, 2])
    static_edge_attr_norm = torch.stack([length_n, speed_n, lanes_n, cycle_len, green_rat], dim=1)  # (E,5)

    # node feature 也可正規化（目前 3 維可先維持原樣）
    return {
        "data_static": data_static,                # 含 x / edge_index / edge_attr(靜態)
        "edge_ids": edge_ids,
        "eid2eidx": eid2eidx,
        "nid2idx": nid2idx,
        "adj": adj,
        "edge_len": edge_len,
        "static_edge_attr_norm": static_edge_attr_norm,  # (E,5) - 固定重用
        "edge_mm": {"length": len_mm, "speed": spd_mm, "lanes": lns_mm}
    }
def collect_dynamic_edge_traveltime(edge_ids, travel_time, MeanSpeed, Occupancy, debug = False):
    # 回傳 shape = (E,) 的 numpy/torch array（與 edge_ids 順序一致）
    cur_time = traci.simulation.getTime()
    vals = []
    for eid in edge_ids:
        try:
            tt = traci.edge.getTraveltime(eid)  # 秒
            tt = travel_time.update(eid, tt, cur_time)
            ms1 = traci.edge.getLastStepMeanSpeed(eid)
            ms2 = MeanSpeed.update(eid, ms1, cur_time)
            occ = traci.edge.getLastStepOccupancy(eid)
            occ = Occupancy.update(eid, occ, cur_time)
            # if debug and eid in ["5707608#2", "5707608#0"]:
            #     print(f"raw ms {ms1}, EMA ms {ms2}")
        except traci.TraCIException:
            tt = np.nan
            ms2 = np.nan
            occ = np.nan
        vals.append([tt, ms2, occ])
    # 簡單處理缺失值
    arr = np.array(vals, dtype=np.float32)
    # sys.exit()
    # if np.isnan(arr).any():
    #     # 用非 NaN 的均值填補
    #     mean_tt = np.nanmean(arr) if not np.isnan(np.nanmean(arr)) else 1.0
    #     arr = np.nan_to_num(arr, nan=float(mean_tt))
    return torch.from_numpy(arr)  # (E,3)

def build_data_runtime(static_pack, travel_time, MeanSpeed, Occupancy, debug = False, edge_feature_mode: str = "all"):
    """
    node: (static)in/out_degree, is_signalized  (dynamic)is_target, recent_flag, visit_age, ever_visit
    edge: (static)length, speed_limit, num_lanes (dynamic)avg_speed, travel_time, occ
    """
    ds = static_pack["data_static"]
    edge_ids = static_pack["edge_ids"]
    static_edge_attr_norm = static_pack["static_edge_attr_norm"]  # (E,5)
    valid_edge_feature_modes = {
        "all",
        "static_only",
        "dynamic_only",
        "travel_time_only",
        "avg_speed_only",
        "occupancy_only",
        "static_travel_time",
        "static_occupancy",
        "static_travel_time_occupancy",
    }
    if edge_feature_mode not in valid_edge_feature_modes:
        raise ValueError(f"Unknown edge_feature_mode: {edge_feature_mode}")

    # 1) 取動態 travel time，並做簡單正規化
    dym = collect_dynamic_edge_traveltime(edge_ids, travel_time, MeanSpeed, Occupancy, debug)  # (E,3)
    dym = torch.nan_to_num(dym, nan=0.0, posinf=0.0, neginf=0.0)
    

    # # 你可以用 min-max 以「當前步」動態縮放，或用移動窗口/全域統計
    # # 這裡先用「當前步 min-max」
    # mn, mx = dym.min(dim=0, keepdim=True).values, dym.max(dim=0, keepdim=True).values
    # dym_norm = (dym - mn) / (mx - mn + 1e-8)
    tt        = dym[:, 0] / 300.0       # 最多 5 分鐘 → 0~1
    avg_speed = dym[:, 1] / 30.0        # 大約 0~1
    occ       = dym[:, 2]               # 已在 0~1

    max_travel_time = 600.0
    max_speed = 30.0
    max_log_tt = torch.log1p(torch.tensor(max_travel_time, dtype=dym.dtype, device=dym.device))
    travel_time_raw = torch.clamp(dym[:, 0], min=0.0, max=max_travel_time)
    avg_speed_raw = torch.clamp(dym[:, 1], min=0.0, max=max_speed)
    occupancy_raw = torch.clamp(dym[:, 2], min=0.0, max=1.0)
    tt = torch.log1p(travel_time_raw) / max_log_tt
    avg_speed = avg_speed_raw / max_speed
    occ = occupancy_raw
    dym_norm = torch.stack([tt, avg_speed, occ], dim=1)  # (E,3)
    dym_norm = torch.nan_to_num(dym_norm, nan=0.0, posinf=1.0, neginf=0.0)
    dym_norm = torch.clamp(dym_norm, min=0.0, max=1.0)
    static_edge_attr_norm = torch.nan_to_num(static_edge_attr_norm, nan=0.0, posinf=1.0, neginf=0.0)
    static_edge_attr_norm = torch.clamp(static_edge_attr_norm, min=0.0, max=1.0)
    def debug_incident_edges(static_pack, dym, dym_norm, edges_to_check, tag=""):
        edge2idx = static_pack.get("edge2idx") or static_pack.get("eid2eidx")
        if edge2idx is None:
            print(f"\n[DEBUG {tag}] edge index mapping not found")
            return
        print(f"\n[DEBUG {tag}] incident edge features")
        for e in edges_to_check:
            if e not in edge2idx:
                print(f"  {e}: NOT IN edge_ids")
                continue
            i = edge2idx[e]
            raw_tt, raw_avg, raw_occ = dym[i].tolist()
            n_tt, n_avg, n_occ = dym_norm[i].tolist()
            print(f"  {e} idx={i:5d} raw(avg={raw_avg:.3f}, tt={raw_tt:.3f}, occ={raw_occ:.3f})"
                  f" | norm(avg={n_avg:.3f}, tt={n_tt:.3f}, occ={n_occ:.3f})")

    if debug:
        debug_incident_edges(
            static_pack, dym, dym_norm,
            edges_to_check=["5707608#2", "5707608#0"],
            tag="after_collect"
        )

    # 2) 合併為固定 schema: [travel_time_norm, length_norm, speed_norm, lanes_norm]
    if edge_feature_mode == "all":
        dynamic_part = dym_norm
        static_part = static_edge_attr_norm
    elif edge_feature_mode == "static_only":
        dynamic_part = torch.zeros_like(dym_norm)
        static_part = static_edge_attr_norm
    elif edge_feature_mode == "dynamic_only":
        dynamic_part = dym_norm
        static_part = torch.zeros_like(static_edge_attr_norm)
    elif edge_feature_mode == "travel_time_only":
        dynamic_part = torch.zeros_like(dym_norm)
        dynamic_part[:, 0] = dym_norm[:, 0]
        static_part = torch.zeros_like(static_edge_attr_norm)
    elif edge_feature_mode == "avg_speed_only":
        dynamic_part = torch.zeros_like(dym_norm)
        dynamic_part[:, 1] = dym_norm[:, 1]
        static_part = torch.zeros_like(static_edge_attr_norm)
    elif edge_feature_mode == "occupancy_only":
        dynamic_part = torch.zeros_like(dym_norm)
        dynamic_part[:, 2] = dym_norm[:, 2]
        static_part = torch.zeros_like(static_edge_attr_norm)
    elif edge_feature_mode == "static_travel_time":
        dynamic_part = torch.zeros_like(dym_norm)
        dynamic_part[:, 0] = dym_norm[:, 0]
        static_part = static_edge_attr_norm
    elif edge_feature_mode == "static_occupancy":
        dynamic_part = torch.zeros_like(dym_norm)
        dynamic_part[:, 2] = dym_norm[:, 2]
        static_part = static_edge_attr_norm
    elif edge_feature_mode == "static_travel_time_occupancy":
        dynamic_part = torch.zeros_like(dym_norm)
        dynamic_part[:, 0] = dym_norm[:, 0]
        dynamic_part[:, 2] = dym_norm[:, 2]
        static_part = static_edge_attr_norm

    edge_attr = torch.cat([dynamic_part, static_part], dim=1)  # (E,8)
    edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1.0, neginf=0.0)
    edge_attr = torch.clamp(edge_attr, min=0.0, max=1.0)

    # 3) node features：先沿用靜態 x（你也可以在這裡拼接動態 node 特徵）
    x = ds.x.clone()
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)

    # 4) 組 Data
    data_t = Data(
        x=x,
        edge_index=ds.edge_index,
        edge_attr=edge_attr
    )
    return data_t  # 這就是每個決策步要丟進 GNN 的資料



def build_dynamic_features(edge_ids):
    subs = traci.edge.getAllSubscriptionResults()
    dyn = []
    for eid in edge_ids:
        r = subs.get(eid, {})
        dyn.append([
            r.get(tc.LAST_STEP_MEAN_SPEED, 0.0),
            r.get(tc.LAST_STEP_OCCUPANCY, 0.0),
            r.get(tc.LAST_STEP_VEHICLE_NUMBER, 0.0),
        ])
    return torch.tensor(np.array(dyn), dtype=torch.float32)   # (E, 3)

#（可選）node 的 TLS 動態：phase、剩餘時間、排隊等
def tls_node_features(nid):
    pass
    # 若該 junction 有號誌，取 phase 與 nextSwitch；沒有則給 0
    # 你也可把「所有 incoming edges 的 waitingTime 合計」當做“排隊長度”
    # return np.array([phase_norm, time_to_switch_norm, queue_norm], dtype=np.float32)


def reachable_edges_from(cur_edge):
    # 回傳從 cur_edge “實際可連”的 next edge 集合（用 lane 連結）
    nlanes = traci.edge.getLaneNumber(cur_edge)
    cur_lanes = [f"{cur_edge}_{i}" for i in range(nlanes)]
    reach = set()
    for lid in cur_lanes:
        for link in traci.lane.getLinks(lid, extended=True):
            to_edge = link[0].split("_")[0]
            reach.add(to_edge)
    return reach

def candidate_next_junctions(net, cur_edge):
    # 幾何上的所有外向 edge
    to_node = net.getEdge(cur_edge).getToNode()
    outs = [e for e in to_node.getOutgoing() if is_external_edge(e)]
    # 實際可走的 edge
    reach = reachable_edges_from(cur_edge)
    outs = [e for e in outs if e.getID() in reach]
    # 轉成“下一個路口”的集合
    jns = sorted({e.getToNode().getID() for e in outs})
    return jns, outs  # 也回傳 outs 方便落地挑邊

def apply_next_junction(ev_id, chosen_jn, outs):

    # 在 outs 中找「toNode == chosen_jn」的邊，挑一條最合適的（例如 traveltime 最小）
    best_e, best_tt = None, 1e9
    for e in outs:
        if e.getToNode().getID() != chosen_jn: 
            continue
        eid = e.getID()
        tt = traci.edge.getTraveltime(eid)  # 或用 mean speed/occupancy 綜合評分
        if tt < best_tt:
            best_tt, best_e = tt, eid
    
    ### 這邊開始接route
    cur_edge = traci.vehicle.getRoadID(ev_id)
    route = traci.vehicle.getRoute(ev_id)
    # print(route)
    idx = traci.vehicle.getRouteIndex(ev_id)
    # if cur_edge.startswith(":"):
    #     print(f"cur_edge {cur_edge}")
    #     print(f"route {route}")
    #     print(f"idx {idx}")

    if cur_edge.startswith(":"):
        # start_idx = max(0, idx - 1)
        start_idx = idx
        # print(f"start_idx {start_idx}")
    else:
        start_idx = idx

    suffix = list(route[start_idx:])
    # suffix = list(route)
    # if cur_edge.startswith(":"):
    #     print(f"suffix {suffix}")
    #     print(f"best_e {best_e}")
    suffix.append(best_e)
    traci.vehicle.setRoute(ev_id, suffix)  
    # print(suffix)
    # print(traci.vehicle.getRoute(ev_id))
    return best_e
    ### end
    
    cur = traci.vehicle.getRoadID(ev_id)
    print(f"[cur edge] {cur}")
    if cur.startswith(":"):
        print(f"next edge {best_e}")
        print(traci.vehicle.getRoute(ev_id))
    traci.vehicle.setRoute(ev_id, [cur, best_e])  
    return best_e  # ★ 回傳本次動作對應的 outgoing edge


""" dijkstra 部分"""

## 道路長度
# def cost_by_length(EDGE_LEN, eid):
def cost_by_length(eid, net):
    # lane_id0 = eid + "_0"
    e = net.getEdge(eid)
    lane_id0 = e.getLanes()[0].getID()          # 真的 laneID
    return traci.lane.getLength(lane_id0)
    # return EDGE_LEN[eid]  # 公尺

## 預計旅行時間(traci估計)
def cost_by_traveltime(eid, net):
    return traci.edge.getTraveltime(eid)  # 秒


def dijkstra_next_hop(net, adj, cur_edge, goal_node, cost_fn):
    """回傳: next_junction, first_edgeID(從當前路口出發的第一步) """
    # print(f"goal_node {goal_node}")
    # 以「當前外部 edge 的出口路口」當作源點
    src = net.getEdge(cur_edge).getToNode().getID() # src 
    if src == goal_node:
        return goal_node, None  # 已在目標路口

    dist = {src: 0.0}
    prev_node = {}
    prev_edge = {}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)

        if d != dist[u]:
            continue
        if u == goal_node:
            break
        for v, eid in adj.get(u, []):
            w = cost_fn(eid)
            nd = d + w

            if (v not in dist) or (nd < dist[v]):
                dist[v] = nd
                prev_node[v] = u
                prev_edge[v] = eid
                heapq.heappush(pq, (nd, v))

    if goal_node not in prev_node:
        return None, None  # 到不了
    
    # 往回還原，取第一步
    v = goal_node
    while prev_node[v] != src:
        v = prev_node[v]
    first_edge = prev_edge[v]
    next_jn = v
    return next_jn, first_edge


def dijkstra_path(net, adj, cur_edge, goal_node, cost_fn, cur_node = None):
    """
    從「cur_edge 的出口路口」(src) 到 goal_node 的最短路。
    參數
      - net: sumolib 的網路
      - adj: dict[node] -> List[(neighbor_node, edge_id)]  你的鄰接表
      - cur_edge: 目前車子所在的邊 id
      - cur_node: 目前車子前往的節點 id
      - goal_node: 目標路口 id
      - cost_fn(eid): 回傳邊 eid 的代價(時間/長度/你定義的權重)
    回傳
      - path_nodes: [src, ..., goal_node]
      - path_edges: [edge_src_to_n1, edge_n1_to_n2, ...]  (與 path_nodes 對齊)
      - total_cost: dist[goal_node]
      若不可達，回傳 (None, None, None)
    """
    if cur_node:
        src = cur_node
    else:
        src = net.getEdge(cur_edge).getToNode().getID()

    if src == goal_node:
        return [src], [], 0.0

    dist = {src: 0.0}
    prev_node = {}
    prev_edge = {}
    pq = [(0.0, src)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == goal_node:
            break

        for v, eid in adj.get(u, []):
            w = cost_fn(eid, net)
            nd = d + w
            if (v not in dist) or (nd < dist[v]):
                dist[v] = nd
                prev_node[v] = u
                prev_edge[v] = eid
                heapq.heappush(pq, (nd, v))

    if goal_node not in prev_node:
        return None, None, None  # 到不了

    # 回溯整條 path（節點與邊）
    path_nodes = [goal_node]
    path_edges = []
    v = goal_node
    while v != src:
        u = prev_node[v]
        eid = prev_edge[v]
        path_nodes.append(u)
        path_edges.append(eid)
        v = u

    path_nodes.reverse()
    path_edges.reverse()
    total_cost = dist[goal_node]
    return path_nodes, path_edges, total_cost

def validate_route_connectivity(net, route_edges):
    '''Raise before TraCI when a route contains internal or illegally connected edges.'''
    if not route_edges:
        raise ValueError('A replacement route must contain at least one edge')
    for edge_id in route_edges:
        if edge_id.startswith(':'):
            raise ValueError(f'Internal edge {edge_id} cannot be part of a vehicle route')
        net.getEdge(edge_id)
    for left_id, right_id in zip(route_edges, route_edges[1:]):
        left = net.getEdge(left_id)
        # Sharing a junction is not sufficient in SUMO: a turn must have an
        # explicit lane connection. Edge.getOutgoing() is built from those
        # connection definitions and therefore catches forbidden turns.
        outgoing_ids = {edge.getID() for edge in left.getOutgoing().keys()}
        if right_id not in outgoing_ids:
            raise ValueError(f'No legal SUMO connection: {left_id} -> {right_id}')


def find_sumo_route_to_node(ev, net, route_anchor, goal_node):
    '''Return the fastest SUMO-valid route from an external edge to a node.'''
    goal = net.getNode(goal_node)
    vehicle_type = traci.vehicle.getTypeID(ev)
    candidates = []

    # Charging stations are represented by goal nodes in this project, while
    # simulation.findRoute requires a destination edge. Try every external
    # incoming edge and retain the fastest route SUMO can actually drive.
    for target_edge in goal.getIncoming():
        target_id = target_edge.getID()
        if target_id.startswith(':') or target_edge.getFunction() == 'internal':
            continue
        try:
            stage = traci.simulation.findRoute(
                route_anchor,
                target_id,
                vType=vehicle_type,
            )
        except traci.exceptions.TraCIException:
            continue

        route_edges = list(stage.edges)
        if not route_edges or route_edges[0] != route_anchor:
            continue
        try:
            validate_route_connectivity(net, route_edges)
        except ValueError:
            continue
        candidates.append((float(stage.travelTime), len(route_edges), route_edges))

    if not candidates:
        raise ValueError(
            f'No SUMO-valid route from {route_anchor} to goal node {goal_node} '
            f'for vehicle type {vehicle_type}'
        )
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def apply_full_route(ev, cur_edge, path_edges, suffix_keep=None, net=None):
    """
    將「目前所在邊 + 最短路邊」組成 route 丟給 SUMO。
    - suffix_keep: 若設成整數 N，僅下達前 N 條後綴邊 (避免一次太長)。
                   None 則整條都下。
    """
    route_ids = [cur_edge] + (path_edges if suffix_keep is None else path_edges[:suffix_keep])
    if net is not None:
        validate_route_connectivity(net, route_ids)
    # print(traci.vehicle.getRoute(ev))
    # print(cur_edge)
    # print(route_ids)
    # 連通性由你的 adj/net 保證；必要時可先檢查 route 長度 > 1
    traci.vehicle.setRoute(ev, route_ids)

"""
處理決策路段過小，來不及choose action的狀況
"""


def is_decisionable_edge(e):
    try:
        return (not e.isInternal()) and (e.getLength() >= MIN_DECISION_LEN)
    except Exception:
        return False
    
def edge_allows(e, vclass):
    if vclass is None:
        return True
    # sumolib 的 allows 接 lane 為主；這裡簡化為只看第一個 lane
    try:
        ln = e.getLanes()[0]
        return ln.allows(vclass)
    except Exception:
        return True

def expand_to_decision_edges(net, start_edge,
                             max_hops=MAX_SKIP_HOPS,
                             vclass=VCLASS_FILTER):
    """
    從 start_e (Edge 物件，可能是 internal/超短) 開始，往前穿越，
    找到「第一層可決策」的邊集合。
    回傳 list[ (target_node_id, first_real_edge_id, full_prefix_edges) ]
    其中 full_prefix_edges 包含 start_edge_id 以及所有被穿越的邊（不含 first_real_edge 本身）。
    """
    
    start_id = start_edge.getID()
    results = []
    q = deque([(start_edge, [])])  # (當前 Edge 物件, prefix_ids)
    visited  = {start_id}
    hops = 0

    while q and hops <= max_hops:
        hops += 1
        for _ in range(len(q)):
            cur_e, prefix = q.popleft()

            # 若當前邊已可決策，且我們確實穿越過至少一條(=prefix非空)，收斂
            if is_decisionable_edge(cur_e) and len(prefix) > 0:
                tgt = cur_e.getToNode().getID()
                results.append((tgt, cur_e.getID(), prefix.copy()))
                continue

            # 繼續往下一層
            nxt_node = cur_e.getToNode()
            for ne in nxt_node.getOutgoing():
                if vclass is not None and not edge_allows(ne, vclass):
                    continue
                ne_id = ne.getID()
                if ne_id in visited:
                    continue
                visited.add(ne_id)
                q.append((ne, prefix + [cur_e.getID()]))

    # 若完全找不到(例如一路都是 internal 且沒有外側邊)，退而求其次
    if not results:
        tgt = start_edge.getToNode().getID()
        results.append((tgt, start_id, []))

    # 依 first_real_edge 去重
    uniq = {}
    for tgt, first_real, pref in results:
        uniq[first_real] = (tgt, first_real, pref)
    return list(uniq.values())

def build_macro_candidates(net, cur_edge_id, outs_edges):
    """
    outs_edges: List[Edge]  # 你的 candidate_next_junctions 第二個回傳值
    回傳 macros: List[dict]
      - node: 展開後第一個可決策邊的 toNode (作為「下一步節點」)
      - first_edge: 需要 via / setRoute 的第一個可決策邊 ID
      - prefix: 需要先穿越的短邊/內部邊 IDs（可能為空）
    """
    macros = []
    for eout in outs_edges:
        if is_decisionable_edge(eout):
            # eout 本身就可決策，直接當作一個巨集
            macros.append({
                "node": eout.getToNode().getID(),
                "first_edge": eout.getID(),
                "prefix": []
            })
        else:
            # 展開：把 eout 穿越成多個可決策邊
            exps = expand_to_decision_edges(net, eout)
            for tgt_node, first_real, prefix in exps:
                macros.append({
                    "node": tgt_node,
                    "first_edge": first_real,
                    "prefix": prefix
                })
    # 依 first_edge 去重（避免同一條邊重複）
    seen = set()
    uniq_macros = []
    for m in macros:
        key = m["first_edge"]
        if key in seen:
            continue
        seen.add(key)
        uniq_macros.append(m)
    return uniq_macros

def _is_internal_edge(e) -> bool:
    # 兼容各版本：先試 getFunction()，不行就用 id 前綴判斷
    try:
        return e.getFunction() == "internal"
    except Exception:
        return str(e.getID()).startswith(":")
    

def is_all_internal_to(net, cur_edge_id: str, target_node_id: str, max_hops: int = 30):
    """
    回傳 (all_internal: bool, total_internal_len: float)
    all_internal=True 代表：從 cur_edge 的出口路口出發，只走 internal edges 就能到 target_node
    total_internal_len：找到的其中一條 internal-only 路徑的總長度（公尺）
    """
    src_node = net.getEdge(cur_edge_id).getToNode()
    if src_node.getID() == target_node_id:
        return True, 0.0  # 已在目標路口

    q = deque()
    visited = set()

    # 從出口路口的所有 outgoing 邊開始，但只走 internal
    for e in src_node.getOutgoing():
        if _is_internal_edge(e):
            q.append((e, e.getLength()))
            visited.add(e.getID())

    hops = 0
    while q and hops <= max_hops:
        hops += 1
        for _ in range(len(q)):
            e, acc_len = q.popleft()
            nxt_node = e.getToNode()
            if nxt_node.getID() == target_node_id:
                return True, acc_len  # 找到一條完全由 internal edges 構成的路徑

            for ne in nxt_node.getOutgoing():
                # 只允許 internal；一旦碰到外部邊，這條路徑就不是 all-internal 了
                if not _is_internal_edge(ne):
                    continue
                if ne.getID() in visited:
                    continue
                visited.add(ne.getID())
                q.append((ne, acc_len + ne.getLength()))

    # 走不純 internal 到達 target_node
    return False, 0.0

def is_internal_edge(edge):
    # sumolib Edge 物件
    try:
        return edge.getFunction() == "internal"
    except Exception:
        return edge.getID().startswith(":")
    
def inspect_link_api(link):
    methods = [m for m in dir(link) if callable(getattr(link, m)) and not m.startswith("_")]
    attrs   = [a for a in dir(link) if not callable(getattr(link, a)) and not a.startswith("_")]
    print("METHODS:", methods)
    print("ATTRS:", attrs)
    
def _get_lane_from_id_or_edge(net, maybe_lane_or_edge_id):
    """
    兼容 via 回傳 edgeID 或 laneID 的情況：
    - 若是 laneID（帶底線），直接 net.getLane(...)
    - 若是 edgeID（不帶底線），改用 net.getEdge(...).getLanes()[0]
    - 若拿不到，回傳 None
    """
    if not maybe_lane_or_edge_id:
        return None
    s = str(maybe_lane_or_edge_id)
    try:
        # 嘗試當作 laneID
        return net.getLane(s)
    except Exception:
        # 若沒有底線或解析失敗，改當作 edgeID
        try:
            e = net.getEdge(s)
            lanes = e.getLanes()
            return lanes[0] if lanes else None
        except Exception:
            return None

def trace_to_target_edge(net, first_link, target_edge_id):
    """
    從入口 lane 的 link 出發，沿 via 追 internal 鏈。
    若最終出口 edge == target_edge_id，回 (internal_edges, exit_lane)，否則 None。
    適用此 sumolib 版本：link.getToLane(), link.getViaLaneID(), via_lane.getOutgoing()
    """
    internal_edges = []
    cur_link = first_link
    guard = 0
    while True:
        guard += 1
        if guard > 32:
            return None  # 防呆

        exit_lane = cur_link.getToLane()            # 直接拿 Lane 物件 (拿出口lane)
        via_lane_id = cur_link.getViaLaneID()       # 可能是 laneID, 也可能是 edgeID, 或 None

        if not via_lane_id:
            # 沒有 via，表示這一步直接落到 external
            exit_edge = exit_lane.getEdge()

            if is_internal_edge(exit_edge):
                return None
            if exit_edge.getID() == target_edge_id:
                return internal_edges, exit_lane
            else:
                return None
        else:
            # 仍需穿越 internal：把 via 解析成 Lane 物件
            via_lane = _get_lane_from_id_or_edge(net, via_lane_id)
            if via_lane is None:
                return None
            int_edge = via_lane.getEdge()
            
            if not is_internal_edge(int_edge):
                # 異常：via 指到 external
                return None
            internal_edges.append(int_edge.getID())


            # 往下一段 internal 走（通常只有一個 outgoing）
            via_links = via_lane.getOutgoing()
            if not via_links:
                return None
            cur_link = via_links[0]

    

def distance_to_next_external_on(net, veh, target_edge_id):
    """
    計算：從目前所在入口 lane，若選擇走到 target_edge_id，
    則 AC 殘距 + internal 鏈長度 + 目標出口 lane 長度。多條合法鏈取最短。
    回傳： (total, internal_len, exit_lane_len, internal_edges, exit_lane_id) 或 None
    """
    cur_lane_id = traci.vehicle.getLaneID(veh)
    lane = net.getLane(cur_lane_id)

    lane_len = lane.getLength()
    lane_pos = traci.vehicle.getLanePosition(veh)
    ac_remain = max(lane_len - lane_pos, 0.0)


    links = lane.getOutgoing() or []
    best = None
    for lk in links:
        traced = trace_to_target_edge(net, lk, target_edge_id)
        if traced is None:
            continue
        internal_edges, exit_lane = traced
        internal_len = sum(net.getEdge(eid).getLength() for eid in internal_edges)
        exit_len = exit_lane.getLength()
        total = ac_remain + internal_len + exit_len
        if (best is None) or (total < best[0]):
            best = (total, internal_len, exit_len, internal_edges, exit_lane.getID())
    return best


def remaining_routes_len(net, veh):
    cur_edge = traci.vehicle.getRoadID(veh)
    route = traci.vehicle.getRoute(veh)
    idx = traci.vehicle.getRouteIndex(veh)
    suffix = list(route[idx:])

    hops = internal_chains_for_route(net, suffix)
    # print(f"hops {hops}")
    ### 找到remaining route
    # 1. remaining route只剩一個 且 cur_edge就是remaining route
    if len(suffix) == 1 :
        if suffix[-1] != cur_edge:
            print("[Error] routes mistake")
            sys.exit()
        lane = traci.vehicle.getLaneID(veh)
        remaining = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(veh)
        return remaining

    # 2. remaining route不只一個 且 cur edge是external
    if not cur_edge.startswith(":"):
        remaining = 0
        cur_edge_remaining = net.getEdge(cur_edge).getLength() - traci.vehicle.getLanePosition(veh)
        remaining += cur_edge_remaining
        for h in hops:
            inter_len = h['internal_len']
            exter_len = net.getEdge(h['to_edge']).getLength()
            remaining += inter_len + exter_len

        return remaining
    # 3. remaining route不只一個 且 cur edge是internal
    if cur_edge.startswith(":"):
        # cur_edge會在第一個hop裡面
        cur_hop = hops[0]
        inter_edges = cur_hop["internal_edges"]
        exit_lane_id = cur_hop["exit_lane_id"]
        idx = inter_edges.index(cur_edge)
        lane_id = traci.vehicle.getLaneID(veh)      # 當前 lane ID
        lane_pos = traci.vehicle.getLanePosition(veh)   # 車子在該 lane 上的位置 (m)

        # 當前 internal 剩餘
        cur_rem = net.getLane(lane_id).getLength() - lane_pos
        # 後續 internal 還要走完
        rest_internal = sum(net.getEdge(eid).getLength() for eid in inter_edges[idx+1:])
        rem_exit = net.getLane(exit_lane_id).getLength()
        remaining = cur_rem + rest_internal + rem_exit
        
        for h in hops[1:]:
            inter_len = h['internal_len']
            exter_len = net.getEdge(h['to_edge']).getLength()
            remaining += inter_len + exter_len
        
        return remaining
    
    print("[Error] else")
    sys.exit()
    return 0.0

def remaining_to_exit_external_end(net, veh, internal_edges, exit_lane_id):
    """
    回傳：從目前位置到「exit_external_edge 的尾端」的剩餘距離（m）
    會自動判斷：在 external gateway / internal / exit external
    """
    exit_lane = net.getLane(exit_lane_id)
    exit_edge = exit_lane.getEdge()
    exit_edge_id = exit_edge.getID()
    
    road = traci.vehicle.getRoadID(veh)     # edgeID (含 ":" 代表 internal)
    lane_id = traci.vehicle.getLaneID(veh)
    lane_pos = traci.vehicle.getLanePosition(veh)
    # print(f"     [remaining_to_exit_external_end], aiming lane {exit_lane_id}, on edge {exit_edge_id}; cur edge {road}")

    # 1) 還在當前 external gateway 或是 已經到exit_lane_id了
    if not road.startswith(":"):
        if road != exit_edge_id: # 還在當前external gateway
            # print(f"     in external gateway")
            rem_gateway = net.getLane(lane_id).getLength() - lane_pos
            # internal_edges 全部 + 完整的 exit external（整條 exit_lane）
            rem_internal = sum(net.getEdge(eid).getLength() for eid in internal_edges)
            rem_exit = net.getLane(exit_lane_id).getLength()
            return max(rem_gateway, 0.0) + rem_internal + rem_exit
        
        else: 
            # 已經在exit_lane_id上了 回傳現在距離底多遠
            # print(f"     in exit_lane")
            rem_gateway = net.getLane(lane_id).getLength() - lane_pos
            return rem_gateway

    

    # 2) 已在 internal：找到當前 internal 在 internal_edges 中的 index
    try:
        idx = internal_edges.index(road)
    except ValueError:
        # 如果不是這條鏈上的 internal 代表現在決定的edge是前一個junction的internal edge

        print('[Error] wrong internal')
        print(f"goal edge {exit_lane_id}")
        cur_edge = traci.vehicle.getRoadID(veh)
        route = traci.vehicle.getRoute(veh)
        idx = traci.vehicle.getRouteIndex(veh)
        print(f"cur_edge {cur_edge}")
        print(f"route {route}")
        print(f"idx {idx}")
        suffix = list(route[idx:])
        print(f"suffix route {suffix}")

        hops = internal_chains_for_route(net, suffix)
        print(f"hops {hops}")
        ### 找到remaining route
        
        sys.exit()
        return 0.0
    

    # 當前 internal 剩餘
    cur_rem = net.getLane(lane_id).getLength() - lane_pos
    # 後續 internal 還要走完
    rest_internal = sum(net.getEdge(eid).getLength() for eid in internal_edges[idx+1:])
    rem_exit = net.getLane(exit_lane_id).getLength()
    return max(cur_rem, 0.0) + rest_internal + rem_exit


def _safe_get_lane(net, lane_or_edge_id: str):
    """允許傳 laneID('edge_0') 或 edgeID('edge')；edgeID 取第0條lane。"""
    s = str(lane_or_edge_id)
    if "_" in s:
        return net.getLane(s)
    e = net.getEdge(s)
    lanes = e.getLanes()
    return lanes if lanes else None

def distance_from_exit_lane_to_target_edge(net, start_exit_lane_id, target_edge_id):
    """
    從【指定的外部出口 lane】出發，穿越下一個路口，
    若能到達 target_edge_id，回傳
      (total, internal_len, exit_lane_len, internal_edges, target_exit_lane_id)
    否則回傳 None。

    注意：這裡不包含「起點 lane 的殘距」，因為起點就是完整外部 lane 的起點語義。
    """
    start_lane = _safe_get_lane(net, start_exit_lane_id)
    for slane in start_lane:
        links = slane.getOutgoing() or []
        best = None

        for lk in links:
            traced = trace_to_target_edge(net, lk, target_edge_id)
            if traced is None:
                continue
            internal_edges, exit_lane = traced
            internal_len = sum(net.getEdge(eid).getLength() for eid in internal_edges)
            exit_len = exit_lane.getLength()
            total = internal_len + exit_len
            if (best is None) or (total < best[0]):
                best = (total, internal_len, exit_len, internal_edges, exit_lane.getID())
    return best


# 2) 在 from_edge → to_edge 之間找 internal 鏈（多條候選取最短）
def internal_chain_between_edges(net, from_edge_id: str, to_edge_id: str):
    from_edge = net.getEdge(from_edge_id)
    best = None
    for lane in from_edge.getLanes():
        for lk in (lane.getOutgoing() or []):
            traced = trace_to_target_edge(net, lk, to_edge_id)
            if traced is None:
                continue
            internal_edges, exit_lane = traced
            internal_len = sum(net.getEdge(eid).getLength() for eid in internal_edges)
            # 取 internal 鏈最短；你也可以加 tie-break（例如出口 lane 指數、方向等）
            if (best is None) or (internal_len < best[0]):
                best = (internal_len, internal_edges, exit_lane.getID())
    if best is None:
        return None  # from→to 之間（在當前 lane 拓撲下）沒有連通
    internal_len, internal_edges, exit_lane_id = best
    return {
        "from_edge": from_edge_id,
        "to_edge": to_edge_id,
        "internal_edges": internal_edges,   # list[str]
        "internal_len": internal_len,       # float (m)
        "exit_lane_id": exit_lane_id,       # 最後落到 to_edge 的哪條 lane
    }

# 3) 對整條 route 逐段求 internal 鏈
def internal_chains_for_route(net, route_edges):
    """
    route_edges: 只含外部 edge 的路徑序列（像你給的那串）
    回傳 list[dict]，每段一個 dict（同 internal_chain_between_edges 的格式）。
    若某段找不到，會回 None 並中止或你可選擇跳過。
    """
    hops = []
    for i in range(len(route_edges) - 1):
        frm = route_edges[i]
        to  = route_edges[i+1]
        info = internal_chain_between_edges(net, frm, to)
        if info is None:
            # 這段在目前 lane-link 下不可達（或需要特定車道）
            # 你可選擇 raise、append None、或嘗試別的策略
            return None
        hops.append(info)
    return hops

GREEN_CHARS = set("gG")

def _pick_active_logic(tls):
    """挑當前 program 對應的 Logic。"""
    prog = traci.trafficlight.getProgram(tls)
    logics = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls)
    # 找同 programID 的，找不到就退而求其次拿第一個
    for L in logics:
        try:
            if getattr(L, "programID", None) == prog:
                return L
        except Exception:
            pass
    return logics[0] if logics else None

def green_ratio_for_link_indices(tls_id: str, link_indices: list[int]):
    """
    回傳 (cycle_length, mean_green_ratio)
    對多個 link index 取平均（你也可改成加權平均）
    """
    logic = _pick_active_logic(tls_id)
    phases = logic.getPhases()
    cycle = sum(float(p.duration) for p in phases)
    if cycle == 0:
        return 0.0, 0.0

    ratios = []
    for li in link_indices:
        gsum = 0.0
        for p in phases:
            if p.state[li] in GREEN_CHARS:
                gsum += float(p.duration)
        ratios.append(gsum / cycle)
    mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return cycle, mean_ratio

def lane2edge(lane_id: str) -> str:
    # SUMO慣例：lane_id 形如 "edgeID_0", "edgeID_1"...
    return lane_id.rsplit("_", 1)[0]

def build_inedge_to_tls_and_indices():
    """
    掃一次所有號誌，把 (inEdge) -> {tls_id: [link_indices...]} 建起來，之後查表很快。
    """
    table = defaultdict(lambda: defaultdict(list))
    for tls in traci.trafficlight.getIDList():
        links = traci.trafficlight.getControlledLinks(tls)
        for idx, group in enumerate(links):
            for in_lane, out_lane, via_lane in group:
                in_edge = lane2edge(in_lane)
                table[in_edge][tls].append(idx)
    return table  # dict[inEdge][tls] = [link_idx,...]

def compute_signal_backbone_for_edges(pyg_edge_in_edge_ids, pyg_edge_target_tls_ids):
    """
    對齊 PyG 的每一條 (u->v) 路段邊，產生兩個特徵：
    - cycle_length_at_v
    - mean_green_ratio_for_inEdge_at_v
    這兩個可以直接 concat 到 GNN 的 edge_attr。
    參數:
      pyg_edge_in_edge_ids: 長度 E 的 list[str]，每條 PyG 邊對應的 SUMO 路段ID（inEdge）
      pyg_edge_target_tls_ids: 長度 E 的 list[str|None]，目標節點 v 的 TLS ID（若該路口無號誌則 None）
    回傳:
      cycle_len (E,), green_ratio (E,)
    """
    E = len(pyg_edge_in_edge_ids)
    inedge2tls = build_inedge_to_tls_and_indices()

    cycle_len = [0.0]*E
    green_rat = [1.0]*E  # 無號誌可設成 1.0（視為不受阻）
    for i in range(E):
        in_e = pyg_edge_in_edge_ids[i]
        tls  = pyg_edge_target_tls_ids[i]
        if not tls:
            cycle_len[i] = 0.0
            green_rat[i] = 1.0
            continue
        link_idx_list = inedge2tls.get(in_e, {}).get(tls, [])
        if not link_idx_list:       # 此 inEdge 在該 tls 下沒有連結（可能不受號誌控）
            cycle_len[i] = 0.0
            green_rat[i] = 1.0
            continue
        c, r = green_ratio_for_link_indices(tls, link_idx_list)
        cycle_len[i] = c
        green_rat[i] = r
    return cycle_len, green_rat


def find_tls_and_link_index_for_movement(in_edge_id: str, out_edge_id: str):
    """
    回傳控制 (in_edge -> out_edge) 的號誌 ID 與其在 state 字串中的索引。
    若無號誌控制，回傳 (None, None)。
    """
    for tls in traci.trafficlight.getIDList():
        links = traci.trafficlight.getControlledLinks(tls)  # list of list[(inLane, outLane, viaLane)]
        for idx, link_group in enumerate(links):
            for in_lane, out_lane, via_lane in link_group:
                if in_lane.startswith(in_edge_id) and out_lane.startswith(out_edge_id):
                    return tls, idx
    return None, None

def time_to_green_and_remaining_green(tls, link_idx: int):
    """
    以「目前相位剩餘 + 往後逐相位累加」估算
    - time_to_green：距離下次進入綠燈的秒數（若已綠則 0）
    - remaining_green：若目前綠燈，估當前相位剩餘綠燈秒數；否則為 0
    """
    now = traci.simulation.getTime()
    cur_idx = traci.trafficlight.getPhase(tls)
    next_sw = traci.trafficlight.getNextSwitch(tls)  # 絕對時間
    rem_cur = max(0.0, next_sw - now)
    logic = _pick_active_logic(tls)
    phases = logic.getPhases()

    cur_char = phases[cur_idx].state[link_idx]
    if cur_char in GREEN_CHARS:
        return 0.0, rem_cur  # 現在就是綠燈

    # 否則從相位尾端開始往後累加，直到遇到某個相位對應 link 為綠
    t = rem_cur
    k = (cur_idx + 1) % len(phases)
    visited = 0
    while visited < len(phases):
        ch = phases[k].state[link_idx]
        if ch in GREEN_CHARS:
            return t, float(phases[k].duration)
        t += float(phases[k].duration)
        k = (k + 1) % len(phases)
        visited += 1
    return float("inf"), 0.0  # 理論上不會發生（代表永不綠）

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
        traci.vehicle.setSpeedMode(ev, 32)  # 權重可依需求
        # print(f"{ev} src {s_edge}, goal node {cs}")





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




def simple_plot(x, xlabel = "Episode", ylabel = "Reward", filename = "tmp123.png"):
    plt.figure()
    plt.plot(x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

def save_routes_xml(
        ev_routes,
        filename,
        start_depart= 13.13,
        depart_gap = 5.0,
        vtype_id = None,
        using_env = " ",
    ):

    filename = f"./{using_env}/new_training_data_3/{filename}_training_set.rou.xml"
    routes = ET.Element(
        "routes",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
        },
    )
    for i, edges in enumerate(ev_routes, start=1):
        veh_id = f"EV_{i}"
        depart_time = start_depart + (i - 1) * depart_gap

        veh_attrs = {
            "id": veh_id,
            "depart": f"{depart_time:.2f}",
            "departLane": "best",
            "departSpeed": "max",
        }
        # 如果你有定義 <vType id="ev" ...>，可以在這裡指定 type
        if vtype_id is not None:
            veh_attrs["type"] = vtype_id   # 對應 <vehicle type="ev">

        veh_elem = ET.SubElement(routes, "vehicle", veh_attrs)

        # 支援：edges 可能是 str 或 list
        if isinstance(edges, (list, tuple)):
            edges_str = " ".join(edges)
        else:
            edges_str = str(edges)

        ET.SubElement(veh_elem, "route", {"edges": edges_str})

    tree = ET.ElementTree(routes)

    # 讓輸出的 XML 有縮排（Python 3.9+）
    try:
        ET.indent(tree, space="    ", level=0)
    except AttributeError:
        # 如果是舊版 Python 沒有 indent，就略過，單行也能用
        pass

    # 寫檔，加上 XML 宣告
    tree.write(filename, encoding="UTF-8", xml_declaration=True)

    return 

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
                except:
                    print(ev)
                    print(start_time[ev])
                    print(goal_nodes[ev])
                    sys.exit()
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

def reached_goal_junction(ev, goal_junc, net, eps=1.0):
    cur_edge_id = traci.vehicle.getRoadID(ev)
    if not cur_edge_id or cur_edge_id.startswith("!"):
        return False  # 尚未上路/瞬間狀態等

    e = net.getEdge(cur_edge_id)

    # 1) 已在 junction 內（internal edge）
    if e.getFunction() == "internal":   # 比 from_node==to_node 更直觀 :contentReference[oaicite:1]{index=1}
        junc_id = e.getFromNode().getID()
        return junc_id == goal_junc

    # 2) 還在一般 edge：快到目標 junction 的停止線/edge 終點
    if e.getToNode().getID() != goal_junc:
        return False

    lane_id = traci.vehicle.getLaneID(ev)
    remain = traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(ev)  # :contentReference[oaicite:2]{index=2}
    return remain <= eps

def freeze_pyg_data(d):
    d = d.clone()  # 先複製 Data 結構 + tensor
    for k in d.keys():
        v = d[k]
        if torch.is_tensor(v):
            d[k] = v.detach().clone().cpu()
        else:
            # 如果你確定沒有非tensor欄位可省略，保守起見 deepcopy
            d[k] = copy.deepcopy(v)
    return d

def action_mask_dqn(net, cur_edge, nid2idx, n=27):
    """
    n: total node num

    return
        可以選的node=1
    """
    jns, outs = candidate_next_junctions(net, cur_edge)
    ###
    # jns: 是to node的id
    # out: 是所有out edge的id
    ###
    out_mask = [nid2idx[no] for no in jns]
    action_mask = torch.zeros(n, device="cpu")
    for i in out_mask:
        action_mask[i] = 1.0
    return action_mask
    sys.exit()

def count_evs_in_rou(rou_path: str, prefix: str = "EV") -> int:
    root = ET.parse(rou_path).getroot()
    n = 0
    for veh in root.findall("vehicle"):
        vid = veh.get("id", "")
        if vid.startswith(prefix):
            n += 1
    return n


def get_evs_in_rou(rou_path: str, prefix: str = "EV") -> list[str]:
    root = ET.parse(rou_path).getroot()
    evs = []
    for veh in root.findall("vehicle"):
        vid = veh.get("id", "")
        if vid.startswith(prefix):
            evs.append(vid)
    return evs


from typing import Dict, List, Tuple, Optional

def _split_edges(edges_str: str) -> List[str]:
    return [e for e in edges_str.strip().split() if e]

def change_route(
    rou_in: str,
    rou_out: str,
    target_vid: str,
    min_edges_left: int = 1,
) -> bool:
    """
    只針對 target_vid，把 <route edges="..."> 的第一個 edge 拿掉，另存成 rou_out。
    回傳 True 代表成功修改；False 代表找不到車或不符合條件（例如 edge 太短）。
    """
    tree = ET.parse(rou_in)
    root = tree.getroot()

    for veh in root.findall("vehicle"):
        if veh.get("id") != target_vid:
            continue

        r = veh.find("route")
        if r is None:
            return False

        edges_str = (r.get("edges") or "").strip()
        edges: List[str] = [e for e in edges_str.split() if e]

        # 至少要能刪掉一個且剩下 min_edges_left
        if len(edges) <= min_edges_left:
            return False

        new_edges = edges[1:]
        if len(new_edges) < min_edges_left:
            return False

        r.set("edges", " ".join(new_edges))
        tree.write(rou_out, encoding="utf-8", xml_declaration=True)
        return True

    return False

import os
import glob
def keep_first_edge_in_rou_file(rou_in: str, rou_out: str) -> tuple[int, int]:
    tree = ET.parse(rou_in)
    root = tree.getroot()
    modified = 0
    skipped = 0

    for veh in root.findall("vehicle"):
        r = veh.find("route")
        if r is None:
            skipped += 1
            continue
        edges_str = (r.get("edges") or "").strip()
        if not edges_str:
            skipped += 1
            continue
        edges = [e for e in edges_str.split() if e]
        if not edges:
            skipped += 1
            continue
        r.set("edges", edges[0])
        modified += 1

    tree.write(rou_out, encoding="utf-8", xml_declaration=True)
    return modified, skipped

def batch_keep_first_edge( pattern: str = "*.rou.xml", suffix: str = "_firstedge", using_env = None, 
                          general_veh_flows = ["free_0900", "norm_1500", "busy_2100", "heavy_2400"]):
    # files = glob.glob(os.path.join(folder, pattern))
    # print(len(files))
    # print(files[0])
    for rd3 in range(100,151):
        for rd4 in range(1,3):
            for rd5 in range(0,4):
                ev_routes = str(f"./{using_env}/training_data_v2/ev_{general_veh_flows[rd5]}_p0{rd4}_{rd3}.rou.xml")
                ev_routes_new = str(f"./{using_env}/new_training_data_2/ev_{general_veh_flows[rd5]}_p0{rd4}_{rd3}.rou.xml")

                keep_first_edge_in_rou_file(ev_routes, ev_routes_new)

    print("done")

    # total_files = 0
    # total_modified = 0
    # total_skipped = 0

    # for f in files:
    #     base, ext = os.path.splitext(f)           # ext = ".xml"
    #     base2, ext2 = os.path.splitext(base)      # ext2 = ".rou"
    #     # 讓輸出長得像 xxx_firstedge.rou.xml
    #     out = f"{base2}{suffix}.rou{ext}"

    #     m, s = keep_first_edge_in_rou_file(f, out)
    #     print(f"✅ {os.path.basename(f)} -> {os.path.basename(out)} | modified={m}, skipped={s}")
    #     total_files += 1
    #     total_modified += m
    #     total_skipped += s

    # print(f"\nDone. files={total_files}, modified_vehicles={total_modified}, skipped_vehicles={total_skipped}")


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
            sys.exit(f"未做選擇已有終點")
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
