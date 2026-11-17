from __future__ import annotations

import traci

from css.routing.topology import is_external_edge
from css.sumo.traffic_signals import lane2edge


def reachable_edges_from(cur_edge):
    # 回傳從 cur_edge “實際可連”的 next edge 集合（用 lane 連結）
    nlanes = traci.edge.getLaneNumber(cur_edge)
    cur_lanes = [f"{cur_edge}_{i}" for i in range(nlanes)]
    reach = set()
    for lid in cur_lanes:
        for link in traci.lane.getLinks(lid, extended=True):
            to_edge = lane2edge(link[0])
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
    return jns, outs


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
    return best_e


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

