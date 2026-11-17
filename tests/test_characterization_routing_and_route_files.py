from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils_sumo import (
    change_route,
    count_evs_in_rou,
    dijkstra_path,
    get_evs_in_rou,
    keep_first_edge_in_rou_file,
    validate_route_connectivity,
)


class FakeEdge:
    def __init__(self, edge_id, length, outgoing=()):
        self.edge_id = edge_id
        self.length = length
        self.outgoing = tuple(outgoing)

    def getID(self):
        return self.edge_id

    def getLength(self):
        return self.length

    def getOutgoing(self):
        return {edge: None for edge in self.outgoing}


class FakeNet:
    def __init__(self, edges):
        self.edges = {edge.getID(): edge for edge in edges}

    def getEdge(self, edge_id):
        return self.edges[edge_id]


class RoutingCharacterizationTests(unittest.TestCase):
    def test_dijkstra_returns_current_node_to_goal_path(self):
        net = FakeNet([FakeEdge("ab", 3.0), FakeEdge("ac", 10.0), FakeEdge("bc", 2.0)])
        adjacency = {"a": [("b", "ab"), ("c", "ac")], "b": [("c", "bc")]}

        nodes, edges, total = dijkstra_path(
            net,
            adjacency,
            cur_edge=None,
            cur_node="a",
            goal_node="c",
            cost_fn=lambda edge_id, graph: graph.getEdge(edge_id).getLength(),
        )

        self.assertEqual(nodes, ["a", "b", "c"])
        self.assertEqual(edges, ["ab", "bc"])
        self.assertEqual(total, 5.0)

    def test_connectivity_rejects_shared_junction_without_legal_turn(self):
        right = FakeEdge("right", 1.0)
        left = FakeEdge("left", 1.0, outgoing=())
        net = FakeNet([left, right])

        with self.assertRaisesRegex(ValueError, "left -> right"):
            validate_route_connectivity(net, ["left", "right"])


class RouteFileCharacterizationTests(unittest.TestCase):
    XML = """<?xml version='1.0'?>
<routes>
  <vehicle id='EV_2'><route edges='a b c'/></vehicle>
  <vehicle id='car_1'><route edges='x y'/></vehicle>
  <vehicle id='EV_1'><route edges='d e'/></vehicle>
</routes>
"""

    def test_ev_count_and_order_follow_xml_order(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "routes.rou.xml"
            source.write_text(self.XML, encoding="utf-8")

            self.assertEqual(count_evs_in_rou(str(source)), 2)
            self.assertEqual(get_evs_in_rou(str(source)), ["EV_2", "EV_1"])

    def test_change_route_drops_only_the_first_edge(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "routes.rou.xml"
            output = Path(directory) / "changed.rou.xml"
            source.write_text(self.XML, encoding="utf-8")

            self.assertTrue(change_route(str(source), str(output), "EV_2"))
            self.assertIn('edges="b c"', output.read_text(encoding="utf-8"))

    def test_keep_first_edge_reports_modified_and_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "routes.rou.xml"
            output = Path(directory) / "first.rou.xml"
            source.write_text(self.XML, encoding="utf-8")

            self.assertEqual(keep_first_edge_in_rou_file(str(source), str(output)), (3, 0))
            text = output.read_text(encoding="utf-8")
            self.assertIn('edges="a"', text)
            self.assertIn('edges="x"', text)
            self.assertIn('edges="d"', text)


if __name__ == "__main__":
    unittest.main()
