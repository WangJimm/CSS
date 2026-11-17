from __future__ import annotations

import sys

import torch
import traci

from css.sumo.routing import candidate_next_junctions


def reached_goal_junction(ev, goal_junc, net, eps=1.0, position_tolerance=1e-3):
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    if position_tolerance < 0:
        raise ValueError(
            f"position_tolerance must be non-negative, got {position_tolerance}"
        )
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
    lane_length = traci.lane.getLength(lane_id)
    lane_position = traci.vehicle.getLanePosition(ev)
    if lane_position > lane_length + position_tolerance:
        return False
    remain = max(lane_length - lane_position, 0.0)
    return remain <= eps


def action_mask_dqn(net, cur_edge, nid2idx, n=None):
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
    if n is None:
        n = len(nid2idx)
    if n <= 0:
        raise ValueError(f"action-mask size must be positive, got {n}")
    out_mask = []
    for node_id in jns:
        if node_id not in nid2idx:
            raise ValueError(
                f"Reachable node {node_id!r} from edge {cur_edge!r} is missing from nid2idx"
            )
        index = nid2idx[node_id]
        if not 0 <= index < n:
            raise ValueError(
                f"Node {node_id!r} maps to index {index}, outside action-mask size {n}"
            )
        out_mask.append(index)
    action_mask = torch.zeros(n, device="cpu")
    for i in out_mask:
        action_mask[i] = 1.0
    return action_mask
