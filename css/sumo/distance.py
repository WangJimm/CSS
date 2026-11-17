from __future__ import annotations

from collections import deque

import traci


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
            raise RuntimeError(
                f"SUMO route mismatch for vehicle {veh!r}: "
                f"current edge is {cur_edge!r}, remaining route is {suffix!r}"
            )
        lane = traci.vehicle.getLaneID(veh)
        remaining = traci.lane.getLength(lane) - traci.vehicle.getLanePosition(veh)
        return max(remaining, 0.0)

    # 2. remaining route不只一個 且 cur edge是external
    if not cur_edge.startswith(":"):
        remaining = 0
        cur_edge_remaining = max(
            net.getEdge(cur_edge).getLength() - traci.vehicle.getLanePosition(veh),
            0.0,
        )
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
        cur_rem = max(net.getLane(lane_id).getLength() - lane_pos, 0.0)
        # 後續 internal 還要走完
        rest_internal = sum(net.getEdge(eid).getLength() for eid in inter_edges[idx+1:])
        rem_exit = net.getLane(exit_lane_id).getLength()
        remaining = cur_rem + rest_internal + rem_exit
        
        for h in hops[1:]:
            inter_len = h['internal_len']
            exter_len = net.getEdge(h['to_edge']).getLength()
            remaining += inter_len + exter_len
        
        return remaining
    
    raise RuntimeError(
        f"Unable to classify current edge {cur_edge!r} for vehicle {veh!r}"
    )


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
            return max(rem_gateway, 0.0)

    

    # 2) 已在 internal：找到當前 internal 在 internal_edges 中的 index
    try:
        idx = internal_edges.index(road)
    except ValueError as exc:
        # 如果不是這條鏈上的 internal 代表現在決定的edge是前一個junction的internal edge
        route = traci.vehicle.getRoute(veh)
        route_idx = traci.vehicle.getRouteIndex(veh)
        suffix = list(route[route_idx:])
        raise RuntimeError(
            f"Vehicle {veh!r} is on internal edge {road!r}, which is not in "
            f"the expected chain {list(internal_edges)!r} for exit lane "
            f"{exit_lane_id!r}; remaining route is {suffix!r}"
        ) from exc
    

    # 當前 internal 剩餘
    cur_rem = net.getLane(lane_id).getLength() - lane_pos
    # 後續 internal 還要走完
    rest_internal = sum(net.getEdge(eid).getLength() for eid in internal_edges[idx+1:])
    rem_exit = net.getLane(exit_lane_id).getLength()
    return max(cur_rem, 0.0) + rest_internal + rem_exit


def _safe_get_lanes(net, lane_or_edge_id: str):
    """Resolve a lane ID or edge ID into a consistent list of lanes."""
    s = str(lane_or_edge_id)
    try:
        return [net.getLane(s)]
    except Exception:
        try:
            return list(net.getEdge(s).getLanes())
        except Exception:
            return []


_safe_get_lane = _safe_get_lanes


def distance_from_exit_lane_to_target_edge(net, start_exit_lane_id, target_edge_id):
    """
    從【指定的外部出口 lane】出發，穿越下一個路口，
    若能到達 target_edge_id，回傳
      (total, internal_len, exit_lane_len, internal_edges, target_exit_lane_id)
    否則回傳 None。

    注意：這裡不包含「起點 lane 的殘距」，因為起點就是完整外部 lane 的起點語義。
    """
    start_lanes = _safe_get_lanes(net, start_exit_lane_id)
    best = None
    for slane in start_lanes:
        links = slane.getOutgoing() or []
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
