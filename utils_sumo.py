"""Deprecated compatibility facade for the reorganized :mod:`css` package."""

import copy
import glob
import heapq
import json
import os
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import traci
from torch_geometric.data import Data
from traci import constants as tc

from css.debug import save_memory_to_txt
from css.experiment.departures import (
    choose_action_random,
    compress_route_on_depart,
    get_dis_from_path,
    is_ev,
)
from css.graph.features import (
    build_data_runtime,
    build_dynamic_features,
    collect_dynamic_edge_traveltime,
    dynamic_node_feature_update,
    freeze_pyg_data,
    get_edge_index_v2,
    prepare_static,
    tls_node_features,
)
from css.legacy.dijkstra import dijkstra_sim
from css.reporting import simple_plot
from css.routing import *  # noqa: F401,F403
from css.routing.topology import MAX_SKIP_HOPS, MIN_DECISION_LEN, VCLASS_FILTER
from css.sumo import *  # noqa: F401,F403
from css.sumo.distance import (
    _get_lane_from_id_or_edge,
    _is_internal_edge,
    _safe_get_lane,
    inspect_link_api,
    is_all_internal_to,
    is_internal_edge,
    trace_to_target_edge,
)
from css.sumo.route_files import _split_edges
from css.sumo.traffic_signals import _pick_active_logic


DEPRECATED_FACADE = True
FACADE_REPLACEMENTS = {
    "routing": "css.routing",
    "sumo": "css.sumo",
    "graph": "css.graph",
    "experiment": "css.experiment",
    "reporting": "css.reporting",
}
