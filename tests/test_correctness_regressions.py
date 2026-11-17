from __future__ import annotations

import types
import unittest
from collections import deque

import torch
from torch_geometric.data import Data

from css.experiment import planning
from css.graph import features
from css.sumo import decisions, distance


class FakeNode:
    def __init__(self, node_id, node_type="priority"):
        self.node_id = node_id
        self.node_type = node_type
        self.incoming = []
        self.outgoing = []

    def getID(self):
        return self.node_id

    def getType(self):
        return self.node_type

    def getIncoming(self):
        return self.incoming

    def getOutgoing(self):
        return self.outgoing


class FakeGraphLane:
    def __init__(self, lane_id):
        self.lane_id = lane_id

    def getID(self):
        return self.lane_id


class FakeGraphEdge:
    def __init__(self, edge_id, source, target, *, length=10.0):
        self.edge_id = edge_id
        self.source = source
        self.target = target
        self.length = length
        self.lanes = [FakeGraphLane(f"{edge_id}_0")]

    def getID(self):
        return self.edge_id

    def getFromNode(self):
        return self.source

    def getToNode(self):
        return self.target

    def getLength(self):
        return self.length

    def getSpeed(self):
        return 10.0

    def getLanes(self):
        return self.lanes

    def getFunction(self):
        return "normal"


class FakeGraphNet:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def getNodes(self):
        return self.nodes

    def getEdges(self):
        return self.edges


class GraphCorrectnessTests(unittest.TestCase):
    def test_empty_recent_history_decays_without_index_error(self):
        data = Data(
            x=torch.zeros((2, 3)),
            edge_index=torch.tensor([[0], [1]]),
            edge_attr=torch.ones((1, 3)),
        )
        ever = torch.zeros((2, 1))
        age = torch.tensor([[0.5], [0.2]])

        updated, _, _ = features.dynamic_node_feature_update(
            data, target=1, recent_node=deque(), ever_visit=ever, visit_age=age
        )

        torch.testing.assert_close(updated.x[:, 4], torch.zeros(2))
        torch.testing.assert_close(updated.x[:, 5], torch.zeros(2))
        torch.testing.assert_close(updated.x[:, 6], torch.tensor([0.475, 0.19]))

    def test_filtered_nodes_and_edges_use_one_aligned_edge_set(self):
        source = FakeNode("source")
        target = FakeNode("target")
        internal = FakeNode(":internal", "internal")
        kept = FakeGraphEdge("kept", source, target, length=12.0)
        filtered = FakeGraphEdge("filtered", source, internal, length=99.0)
        source.outgoing = [kept, filtered]
        target.incoming = [kept]
        internal.incoming = [filtered]

        data, edge_ids, node_index, adjacency, lengths = features.get_edge_index_v2(
            FakeGraphNet([source, target, internal], [kept, filtered])
        )

        self.assertEqual(node_index, {"source": 0, "target": 1})
        self.assertEqual(edge_ids, ["kept"])
        self.assertEqual(dict(adjacency), {"source": [("target", "kept")]})
        self.assertEqual(lengths, {"kept": 12.0})
        self.assertEqual(data.edge_index.shape[1], data.edge_attr.shape[0])
        self.assertEqual(data.edge_attr.shape[0], len(edge_ids))


class FakeLane:
    def __init__(self, lane_id, length=10.0):
        self.lane_id = lane_id
        self.length = length
        self.outgoing = []

    def getID(self):
        return self.lane_id

    def getLength(self):
        return self.length

    def getOutgoing(self):
        return self.outgoing


class FakeEdge:
    def __init__(self, edge_id, lanes=(), *, target_node="goal", internal=False):
        self.edge_id = edge_id
        self.lanes = list(lanes)
        self.target_node = target_node
        self.internal = internal

    def getID(self):
        return self.edge_id

    def getLanes(self):
        return self.lanes

    def getFunction(self):
        return "internal" if self.internal else "normal"

    def getToNode(self):
        return types.SimpleNamespace(getID=lambda: self.target_node)


class FakeLaneNet:
    def __init__(self, lanes, edges):
        self.lanes = {lane.getID(): lane for lane in lanes}
        self.edges = {edge.getID(): edge for edge in edges}

    def getLane(self, lane_id):
        return self.lanes[lane_id]

    def getEdge(self, edge_id):
        return self.edges[edge_id]


class DistanceCorrectnessTests(unittest.TestCase):
    def test_lane_and_edge_ids_resolve_to_consistent_lists(self):
        lane = FakeLane("road_segment_0")
        edge = FakeEdge("road_segment", [lane])
        net = FakeLaneNet([lane], [edge])

        self.assertEqual(distance._safe_get_lanes(net, "road_segment_0"), [lane])
        self.assertEqual(distance._safe_get_lanes(net, "road_segment"), [lane])
        self.assertEqual(distance._safe_get_lanes(net, "missing"), [])

    def test_target_distance_keeps_best_across_all_source_lanes(self):
        first = FakeLane("start_0")
        second = FakeLane("start_1")
        first.outgoing = ["short"]
        second.outgoing = ["long"]
        target_short = FakeLane("target_0", 5.0)
        target_long = FakeLane("target_1", 10.0)
        edge = FakeEdge("start", [first, second])
        net = FakeLaneNet([first, second], [edge])
        original = distance.trace_to_target_edge
        distance.trace_to_target_edge = lambda _net, link, _target: (
            [], target_short if link == "short" else target_long
        )
        try:
            result = distance.distance_from_exit_lane_to_target_edge(net, "start", "target")
        finally:
            distance.trace_to_target_edge = original

        self.assertEqual(result[0], 5.0)
        self.assertEqual(result[-1], "target_0")

    def test_single_edge_remaining_distance_is_never_negative(self):
        fake_traci = types.SimpleNamespace(
            vehicle=types.SimpleNamespace(
                getRoadID=lambda _ev: "edge",
                getRoute=lambda _ev: ("edge",),
                getRouteIndex=lambda _ev: 0,
                getLaneID=lambda _ev: "edge_0",
                getLanePosition=lambda _ev: 12.0,
            ),
            lane=types.SimpleNamespace(getLength=lambda _lane: 10.0),
        )
        original = distance.traci
        distance.traci = fake_traci
        try:
            self.assertEqual(distance.remaining_routes_len(object(), "EV_0"), 0.0)
        finally:
            distance.traci = original

    def test_route_state_mismatch_raises_contextual_error(self):
        fake_traci = types.SimpleNamespace(
            vehicle=types.SimpleNamespace(
                getRoadID=lambda _ev: "actual-edge",
                getRoute=lambda _ev: ("route-edge",),
                getRouteIndex=lambda _ev: 0,
            )
        )
        original = distance.traci
        distance.traci = fake_traci
        try:
            with self.assertRaisesRegex(RuntimeError, "SUMO route mismatch.*EV_0"):
                distance.remaining_routes_len(object(), "EV_0")
        finally:
            distance.traci = original


class PlanningAndDecisionCorrectnessTests(unittest.TestCase):
    def test_movement_lane_matching_uses_exact_edge_ids(self):
        links = [
            [("edge10_0", "outbound_0", "via")],
            [("edge1_0", "out_0", "via")],
        ]
        fake = types.SimpleNamespace(
            trafficlight=types.SimpleNamespace(
                getIDList=lambda: ["tls"],
                getControlledLinks=lambda _tls: links,
            )
        )
        original = planning.traci
        planning.traci = fake
        try:
            self.assertEqual(planning.resolve_movement_out_lanes("edge1", "out"), ["out_0"])
        finally:
            planning.traci = original

    def test_one_hot_requires_explicit_valid_dimension(self):
        torch.testing.assert_close(planning.oneHot(1, 3), torch.tensor([0.0, 1.0, 0.0]))
        with self.assertRaisesRegex(ValueError, "outside"):
            planning.oneHot(3, 3)

    def test_action_mask_derives_size_and_validates_topology(self):
        original = decisions.candidate_next_junctions
        decisions.candidate_next_junctions = lambda _net, _edge: (["b"], [])
        try:
            torch.testing.assert_close(
                decisions.action_mask_dqn(object(), "edge", {"a": 0, "b": 1}),
                torch.tensor([0.0, 1.0]),
            )
            with self.assertRaisesRegex(ValueError, "missing from nid2idx"):
                decisions.action_mask_dqn(object(), "edge", {"a": 0})
        finally:
            decisions.candidate_next_junctions = original

    def test_goal_reached_rejects_large_position_overshoot(self):
        edge = FakeEdge("edge", target_node="goal")
        net = types.SimpleNamespace(getEdge=lambda _edge_id: edge)
        position = {"value": 100.0005}
        fake = types.SimpleNamespace(
            vehicle=types.SimpleNamespace(
                getRoadID=lambda _ev: "edge",
                getLaneID=lambda _ev: "edge_0",
                getLanePosition=lambda _ev: position["value"],
            ),
            lane=types.SimpleNamespace(getLength=lambda _lane: 100.0),
        )
        original = decisions.traci
        decisions.traci = fake
        try:
            self.assertTrue(decisions.reached_goal_junction("EV_0", "goal", net))
            position["value"] = 101.0
            self.assertFalse(decisions.reached_goal_junction("EV_0", "goal", net))
        finally:
            decisions.traci = original


if __name__ == "__main__":
    unittest.main()
