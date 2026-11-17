from __future__ import annotations

import heapq

import traci


def cost_by_length(eid, net):
    # lane_id0 = eid + "_0"
    e = net.getEdge(eid)
    lane_id0 = e.getLanes()[0].getID()          # 真的 laneID
    return traci.lane.getLength(lane_id0)


def cost_by_traveltime(eid, net):
    return traci.edge.getTraveltime(eid)


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

