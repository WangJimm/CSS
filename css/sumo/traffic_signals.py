from __future__ import annotations

from collections import defaultdict

import traci

GREEN_CHARS = set("gG")


def _pick_active_logic(tls):
    """挑當前 program 對應的 Logic。"""
    prog = traci.trafficlight.getProgram(tls)
    logics = traci.trafficlight.getAllProgramLogics(tls)
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
    return table


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
    return float("inf"), 0.0
