from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np
import torch
import traci
from torch_geometric.data import Data
from traci import constants as tc

from css.routing.topology import is_external_edge, is_external_node
from css.sumo.traffic_signals import compute_signal_backbone_for_edges, build_inedge_to_tls_and_indices

DECAY = 0.95
FLOOR = 0


def dynamic_node_feature_update(data_static, target, recent_node, ever_visit, visit_age):
    """
    更新data_static的動態node feature部分
    target: 用的是轉換後的編號
    recent_node: deque, 最近幾次拜訪過的點
    ever_visit: torch [node_len, 1], 曾經拜訪過的點 
    visit_age: torch [node_len, 1], 最近拜訪的點，越近拜訪分數越大
    """

    is_target = torch.zeros(
        (len(data_static.x), 1),
        dtype=data_static.x.dtype,
        device=data_static.x.device,
    )
    is_target[target] = 1

    recent_flag = torch.zeros_like(is_target)
    for nid in recent_node:
        recent_flag[nid] = 1.0

    visit_age.mul_(DECAY).clamp_(min=FLOOR)
    if recent_node:
        latest_node = recent_node[-1]
        ever_visit[latest_node] = 1
        visit_age[latest_node] = 1.0

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
    graph_edges = [
        edge
        for edge in ext_edges
        if edge.getFromNode().getID() in nid2idx
        and edge.getToNode().getID() in nid2idx
    ]


    # node features（示例：是否有號誌、入/出度）
    Xn = np.zeros((len(nodes), 3), np.float32)
    for n in nodes:
        i = nid2idx[n.getID()]
        tls = 1.0 if n.getType().startswith("traffic_light") else 0.0
        indeg, outdeg = len([e for e in n.getIncoming() if e.getFunction() != "internal"]), len([e for e in n.getOutgoing() if e.getFunction() != "internal"])
        Xn[i] = [tls, indeg, outdeg]

    
    # edge_index 與靜態 edge_attr（長度、限速、車道數）
    u, v, Xe = [], [], []
    for e in graph_edges:
        fr, to = e.getFromNode().getID(), e.getToNode().getID()
        u.append(nid2idx[fr]); v.append(nid2idx[to])
        Xe.append([e.getLength(), e.getSpeed(), len(e.getLanes())])

    edge_index = torch.tensor([u, v], dtype=torch.long)
    data_static = Data(
        x=torch.from_numpy(Xn),
        edge_index=edge_index,
        edge_attr=torch.tensor(np.asarray(Xe, dtype=np.float32).reshape(-1, 3)),
    )
    # 保留對應，之後動態取值要用
    edge_ids = [e.getID() for e in graph_edges]
    # print(edge_index)

    adj = defaultdict(list) # adj紀錄每個edge的長度
    edge_len = {}
    for e in graph_edges:
        # e = <edge id="A0A1" from="A0" to="A1"/>
        u = e.getFromNode().getID()
        v = e.getToNode().getID()
        # u = A0, v = A1
        eid = e.getID()
        # eid = A0A1
        edge_len[eid] = e.getLength() # 可以直接獲取該路段長度
        adj[u].append((v, eid))
    
    return data_static, edge_ids, nid2idx, adj, edge_len


def prepare_static(net, *, prefer_controlled_tls=False):
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
    controlled = build_inedge_to_tls_and_indices() if prefer_controlled_tls else {}
    pyg_edge_in_edge_ids = edge_ids
    pyg_edge_target_tls_ids = []
    for e in pyg_edge_in_edge_ids:
        v = net.getEdge(e).getToNode().getID()
        controllers = controlled.get(e, {})
        if controllers:
            pyg_edge_target_tls_ids.append(v if v in controllers else sorted(controllers)[0])
        elif v in tls_pool:
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
    return torch.from_numpy(arr)


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
    return data_t


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
    return torch.tensor(np.array(dyn), dtype=torch.float32)


def tls_node_features(nid):
    raise NotImplementedError(
        "tls_node_features() has no defined feature schema; "
        "use prepare_static() and dynamic_node_feature_update() instead"
    )


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
