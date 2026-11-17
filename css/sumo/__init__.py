"""SUMO and TraCI integration operations."""

from .decisions import action_mask_dqn, reached_goal_junction
from .distance import (
    distance_from_exit_lane_to_target_edge,
    distance_to_next_external_on,
    internal_chain_between_edges,
    internal_chains_for_route,
    remaining_routes_len,
    remaining_to_exit_external_end,
)
from .route_files import (
    batch_keep_first_edge,
    change_route,
    count_evs_in_rou,
    get_evs_in_rou,
    keep_first_edge_in_rou_file,
    save_routes_xml,
)
from .routing import (
    apply_full_route,
    apply_next_junction,
    candidate_next_junctions,
    find_sumo_route_to_node,
    reachable_edges_from,
    validate_route_connectivity,
)
from .spawning import spawn_evs, spawn_evs_testing, spawn_for_rl
from .traffic_signals import (
    build_inedge_to_tls_and_indices,
    compute_signal_backbone_for_edges,
    find_tls_and_link_index_for_movement,
    green_ratio_for_link_indices,
    lane2edge,
    time_to_green_and_remaining_green,
)

__all__ = [
    "action_mask_dqn",
    "apply_full_route",
    "apply_next_junction",
    "batch_keep_first_edge",
    "build_inedge_to_tls_and_indices",
    "candidate_next_junctions",
    "change_route",
    "compute_signal_backbone_for_edges",
    "count_evs_in_rou",
    "distance_from_exit_lane_to_target_edge",
    "distance_to_next_external_on",
    "find_sumo_route_to_node",
    "find_tls_and_link_index_for_movement",
    "get_evs_in_rou",
    "green_ratio_for_link_indices",
    "internal_chain_between_edges",
    "internal_chains_for_route",
    "keep_first_edge_in_rou_file",
    "lane2edge",
    "reachable_edges_from",
    "reached_goal_junction",
    "remaining_routes_len",
    "remaining_to_exit_external_end",
    "save_routes_xml",
    "spawn_evs",
    "spawn_evs_testing",
    "spawn_for_rl",
    "time_to_green_and_remaining_green",
    "validate_route_connectivity",
]
