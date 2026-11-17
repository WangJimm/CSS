"""Pure network topology and shortest-path operations."""

from .algorithms import cost_by_length, cost_by_traveltime, dijkstra_next_hop, dijkstra_path
from .topology import (
    build_macro_candidates,
    edge_allows,
    expand_to_decision_edges,
    is_decisionable_edge,
    is_external_edge,
    is_external_node,
)

__all__ = [
    "build_macro_candidates",
    "cost_by_length",
    "cost_by_traveltime",
    "dijkstra_next_hop",
    "dijkstra_path",
    "edge_allows",
    "expand_to_decision_edges",
    "is_decisionable_edge",
    "is_external_edge",
    "is_external_node",
]
