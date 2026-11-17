from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import traci
from sumolib.net import readNet

from css.experiment.config import EMA_feature
from css.graph import build_data_runtime, prepare_static


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a short SUMO/package integration smoke test.")
    parser.add_argument("--sumo", type=Path, default=Path(os.environ.get("SUMO_BINARY", "sumo")))
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    environment = PROJECT_ROOT / "env"
    config = environment / "osm.sumocfg"
    net_path = environment / "env_static_tls.net.xml"
    general = environment / "training_data" / "output_1800_1.rou.xml"
    ev_routes = environment / "training_data" / "ev_1800_p01_1.rou.xml"
    command = [
        str(args.sumo),
        "-c",
        str(config),
        "--route-files",
        f"{general},{ev_routes}",
        "--no-step-log",
        "--no-warnings",
    ]

    traci.start(command)
    try:
        net = readNet(str(net_path), withInternal=True)
        static_pack = prepare_static(net)
        for _ in range(args.steps):
            traci.simulationStep()
        travel_time = EMA_feature(half_life_sec=10.0)
        mean_speed = EMA_feature(half_life_sec=10.0)
        occupancy = EMA_feature(half_life_sec=10.0)
        runtime = build_data_runtime(static_pack, travel_time, mean_speed, occupancy)
        result = {
            "nodes": int(runtime.x.size(0)),
            "edges": int(runtime.edge_index.size(1)),
            "edge_feature_rows": int(runtime.edge_attr.size(0)),
            "edge_feature_columns": int(runtime.edge_attr.size(1)),
        }
        print(json.dumps(result, sort_keys=True))
        if result["edges"] != result["edge_feature_rows"]:
            raise RuntimeError(f"Graph edge alignment failed: {result}")
    finally:
        traci.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
