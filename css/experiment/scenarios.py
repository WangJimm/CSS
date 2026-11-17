from __future__ import annotations

import json
import csv
from functools import lru_cache
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from css.experiment.config import (
    RUN_DIR,
    TRAINING_SEED_END,
    TRAINING_SEED_START,
    cfg_path,
    testing_veh_flows,
    training_veh_flows,
)
from css.experiment.environments import EnvironmentConfig
from css.sumo.route_files import count_evs_in_rou, get_evs_in_rou

@dataclass
class SumoLoadConfig:
    rd1: int
    rd2: int
    rd3: int
    veh_flow: str | int
    general: str
    ev_routes: str
    evs: List[str]
    load_args: List[str]
    # goal_nodes: Dict[str, str]   # 或 str，看你 goal node 型別
    ev_power : Dict[str, float]
    split: str = ""


@dataclass(frozen=True)
class ScenarioKey:
    scenario_index: int
    probability_index: int
    seed: int


@lru_cache(maxsize=8)
def _manifest_rows(root: str):
    with (Path(root) / "scenario_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    seen = set()
    seed_splits = {}
    for row in rows:
        key = (row["split"], int(row["flow_veh_h"]), int(row["penetration"][1:]), int(row["background_seed"]))
        if key in seen:
            raise ValueError(f"Duplicate manifest scenario: {key}")
        seen.add(key)
        if key[3] in seed_splits and seed_splits[key[3]] != key[0]:
            raise ValueError(f"Generator seed occurs in multiple dataset splits: {key[3]}")
        seed_splits[key[3]] = key[0]
        for field in ("general_route_file", "ev_route_file"):
            path = (Path(root) / row[field]).resolve()
            if not path.is_relative_to(Path(root).resolve()) or path.parent != (Path(root) / f'{row["split"]}_data').resolve():
                raise ValueError(f"Invalid manifest path: {row[field]}")
    return rows


def scenario_keys_for_split(environment, split, veh_flows=None, flow_indices=None, probability_indices=None):
    flows = list(veh_flows) if veh_flows is not None else list(environment.flows(split))
    indices = list(range(len(flows))) if flow_indices is None else list(flow_indices)
    probabilities = [1, 2] if probability_indices is None else list(probability_indices)
    if not indices or not probabilities or any(i < 0 or i >= len(flows) for i in indices):
        raise ValueError("Scenario filters must select valid flow and probability indices")
    if environment.manifest_based:
        keys = [ScenarioKey(flows.index(int(row["flow_veh_h"])), int(row["penetration"][1:]), int(row["background_seed"]))
                for row in _manifest_rows(str(environment.root))
                if row["split"] == split and int(row["flow_veh_h"]) in flows
                and flows.index(int(row["flow_veh_h"])) in indices and int(row["penetration"][1:]) in probabilities]
    else:
        seeds = range(TRAINING_SEED_START, TRAINING_SEED_END + 1) if split == "training" else range(51, 61)
        keys = [ScenarioKey(i, p, seed) for seed in seeds for p in probabilities for i in indices]
    if not keys:
        raise ValueError(f"No scenarios selected for {environment.name}/{split}")
    return sorted(keys, key=lambda k: (k.scenario_index, k.probability_index, k.seed)) if environment.manifest_based else keys


def _load_manifest_scenario(environment, split, key, flows):
    flow = _resolve_veh_flow(flows, key.scenario_index)
    matches = [row for row in _manifest_rows(str(environment.root)) if row["split"] == split
               and int(row["flow_veh_h"]) == int(flow) and int(row["penetration"][1:]) == key.probability_index
               and int(row["background_seed"]) == key.seed]
    if len(matches) != 1:
        raise ValueError(f"Scenario is not in {environment.name}/{split}: flow={flow}, p={key.probability_index}, seed={key.seed}")
    row = matches[0]
    general, routes = environment.root / row["general_route_file"], environment.root / row["ev_route_file"]
    power = environment.root / f'{split}_data_power' / f'power_ev_{flow}_p0{key.probability_index}_{key.seed}.json'
    _require_files(general)
    evs, powers = _load_ev_power_mapping(routes, power)
    if len(evs) != int(row["ev_vehicle_count"]):
        raise ValueError(f"Manifest EV count mismatch: {routes}")
    args = ["-c", str(environment.cfg_path), "--step-length", "0.5", "--route-files", f"{general},{routes}", "--no-warnings"]
    return SumoLoadConfig(key.seed, key.probability_index, key.scenario_index, flow, str(general), str(routes), evs, args, powers, split)


def validate_environment_assets(environment):
    """Read all selected inputs before starting SUMO; never launch a simulation."""
    _require_files(environment.cfg_path, environment.net_path)
    import xml.etree.ElementTree as ET
    nodes = {n.get("id") for n in ET.parse(environment.net_path).getroot().findall("junction")}
    if any(station not in nodes for station in environment.stations):
        raise ValueError(f"Charging stations are missing from {environment.net_path}")
    if not environment.manifest_based:
        return {"environment": environment.name, "data_root": str(environment.root), "manifest_based": False}
    counts = {}
    powers_count = 0
    for split in ("training", "validation", "testing"):
        keys = scenario_keys_for_split(environment, split)
        expected = 420 if split == "training" else 90
        if len(keys) != expected:
            raise ValueError(f"Expected {expected} scenarios in {split}, found {len(keys)}")
        for key in keys:
            cfg = _load_manifest_scenario(environment, split, key, list(environment.flows(split)))
            goal = environment.root / f'{split}_data_goal' / f'goal_ev_{cfg.veh_flow}_p0{key.probability_index}_{key.seed}.json'
            _require_files(goal)
            goals = json.loads(goal.read_text(encoding="utf-8"))
            if not isinstance(goals, list) or len(goals) != len(cfg.evs) or any(g not in environment.stations for g in goals):
                raise ValueError(f"Goal file does not match EVs or stations: {goal}")
            if any(not .2 <= value <= .4 for value in cfg.ev_power.values()):
                raise ValueError(f"{environment.name} SOC outside [0.2, 0.4]: {cfg.ev_routes}")
            powers_count += len(cfg.evs)
        counts[split] = len(keys)
    return {"environment": environment.name, "data_root": str(environment.root), "manifest_based": True,
            "scenario_counts": counts, "ev_power_assignments": powers_count, "charging_stations": list(environment.stations)}


def _default_flow_indices(veh_flows: List[str | int]) -> List[int]:
    return list(range(len(veh_flows)))


def _resolve_veh_flow(veh_flows: List[str | int], flow_index: int) -> str | int:
    try:
        return veh_flows[flow_index]
    except IndexError as exc:
        raise ValueError(
            f"flow_index {flow_index} is out of range for veh_flows={veh_flows}"
        ) from exc


def choose_training_scenario_key(
    rng: random.Random | None = None,
    veh_flows: List[str | int] | None = None,
    flow_indices: List[int] | None = None,
    probability_indices: List[int] | None = None,
    seed_start: int = TRAINING_SEED_START,
    seed_end: int = TRAINING_SEED_END,
    environment: EnvironmentConfig | None = None,
):
    if environment is None:
        from css.experiment.environments import resolve_environment
        environment = resolve_environment()
    rng = rng or random.Random()
    if environment is not None and environment.manifest_based:
        return rng.choice(scenario_keys_for_split(environment, "training", veh_flows, flow_indices, probability_indices))
    veh_flows = veh_flows or training_veh_flows
    flow_indices = flow_indices or _default_flow_indices(veh_flows)
    probability_indices = probability_indices or [1, 2]
    return ScenarioKey(
        scenario_index=rng.choice(flow_indices),
        probability_index=rng.choice(probability_indices),
        seed=rng.randint(seed_start, seed_end),
    )


def build_training_scenario_schedule(
    num_episodes: int,
    schedule_seed: int = 42,
    veh_flows: List[str | int] | None = None,
    flow_indices: List[int] | None = None,
    probability_indices: List[int] | None = None,
    seed_start: int = TRAINING_SEED_START,
    seed_end: int = TRAINING_SEED_END,
    environment: EnvironmentConfig | None = None,
):
    rng = random.Random(schedule_seed)
    return [
        choose_training_scenario_key(
            rng=rng,
            veh_flows=veh_flows,
            flow_indices=flow_indices,
            probability_indices=probability_indices,
            seed_start=seed_start,
            seed_end=seed_end,
            environment=environment,
        )
        for _ in range(num_episodes)
    ]


def _results_output_dir(environment=None):
    output_dir = environment.output_dir if environment is not None else Path(__file__).resolve().parents[2] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _make_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_training_results(payload, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.{os.getpid()}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(payload), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)


def _require_files(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing scenario assets: " + ", ".join(missing))


def _load_ev_power_mapping(ev_routes: Path, power_path: Path) -> tuple[list[str], dict[str, float]]:
    _require_files(ev_routes, power_path)
    evs = get_evs_in_rou(str(ev_routes))
    with power_path.open("r", encoding="utf-8") as stream:
        powers = json.load(stream)
    if not isinstance(powers, list):
        raise ValueError(f"Power data must be a JSON list: {power_path}")
    if len(evs) != len(powers):
        raise ValueError(
            f"EV/power count mismatch for {ev_routes.name}: "
            f"{len(evs)} EVs but {len(powers)} power values in {power_path.name}"
        )
    if len(set(evs)) != len(evs):
        raise ValueError(f"Duplicate EV IDs: {ev_routes}")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in powers):
        raise ValueError(f"Power data must contain finite SOC fractions in [0, 1]: {power_path}")
    return evs, dict(zip(evs, powers))


def require_higher_evaluation_assets(run_dir: Path = RUN_DIR) -> None:
    required_directories = (
        run_dir / "higher_test_data",
        run_dir / "higher_test_data_power",
    )
    missing = [str(path) for path in required_directories if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Higher evaluation was requested, but required directories are missing: "
            + ", ".join(missing)
        )


def load_sumo(
    eval = False,
    rd1 = None,
    rd2 = None,
    rd3 = None,
    scenario_key: ScenarioKey | None = None,
    veh_flows: List[str | int] | None = None,
    rng: random.Random | None = None,
    environment: EnvironmentConfig | None = None,
    split: str | None = None,
):
    if environment is None:
        from css.experiment.environments import resolve_environment
        environment = resolve_environment()
    selected_split = split or ("testing" if eval else "training")
    if environment is not None and environment.manifest_based:
        flows = list(veh_flows) if veh_flows is not None else list(environment.flows(selected_split))
        if scenario_key is None:
            if all(value is not None for value in (rd1, rd2, rd3)):
                scenario_key = ScenarioKey(rd3, rd2, rd1)
            else:
                scenario_key = (rng or random).choice(scenario_keys_for_split(environment, selected_split, flows))
        return _load_manifest_scenario(environment, selected_split, scenario_key, flows)
    run_dir = environment.root if environment is not None else RUN_DIR
    selected_cfg = str(environment.cfg_path) if environment is not None else cfg_path
    if not eval:
        veh_flows = veh_flows or training_veh_flows
        if scenario_key is None:
            rng = rng or random
            rd1 = rng.randint(TRAINING_SEED_START, TRAINING_SEED_END)
            rd2 = rng.randint(1, 2)
            rd3 = rng.randint(0, len(veh_flows) - 1)
        else:
            rd1 = scenario_key.seed
            rd2 = scenario_key.probability_index
            rd3 = scenario_key.scenario_index

        veh_flow = _resolve_veh_flow(veh_flows, rd3)
        general_path = run_dir / f"training_data/output_{veh_flow}_{rd1}.rou.xml"
        ev_routes_path = run_dir / f"training_data/ev_{veh_flow}_p0{rd2}_{rd1}.rou.xml"
        power_path = run_dir / f"training_data_power/power_ev_{veh_flow}_p0{rd2}_{rd1}.json"
    else:
        veh_flows = veh_flows or testing_veh_flows
        veh_flow = _resolve_veh_flow(veh_flows, rd3)
        general_path = run_dir / f"testing_data/output_{veh_flow}_{rd1}.rou.xml"
        ev_routes_path = run_dir / f"testing_data/ev_{veh_flow}_p0{rd2}_{rd1}.rou.xml"
        power_path = run_dir / f"testing_data_power/power_ev_{veh_flow}_p0{rd2}_{rd1}.json"

    _require_files(general_path)
    evs, ev_powers = _load_ev_power_mapping(ev_routes_path, power_path)
    general = str(general_path)
    ev_routes = str(ev_routes_path)
    load_args = ["-c", selected_cfg, "--step-length", "0.5", "--route-files", f"{general}, {ev_routes}", "--no-warnings"]

    return SumoLoadConfig(rd1, rd2, rd3, veh_flow, general, ev_routes, evs, load_args, ev_powers, selected_split)


def load_higher_eval_sumo(rd1, rd2, rd3, veh_flows: List[str | int] | None = None, environment=None):
    run_dir = environment.root if environment is not None else RUN_DIR
    selected_cfg = str(environment.cfg_path) if environment is not None else cfg_path
    require_higher_evaluation_assets(run_dir)
    veh_flows = veh_flows or testing_veh_flows
    veh_flow = _resolve_veh_flow(veh_flows, rd3)
    general_path = run_dir / f"higher_test_data/output_{veh_flow}_{rd1}.rou.xml"
    ev_routes_path = run_dir / f"higher_test_data/ev_{veh_flow}_p0{rd2}_{rd1}.rou.xml"
    power_path = run_dir / f"higher_test_data_power/power_ev_{veh_flow}_p0{rd2}_{rd1}.json"
    _require_files(general_path)
    evs, ev_powers = _load_ev_power_mapping(ev_routes_path, power_path)
    general = str(general_path)
    ev_routes = str(ev_routes_path)
    load_args = ["-c", selected_cfg, "--step-length", "0.5", "--route-files", f"{general}, {ev_routes}", "--no-warnings"]

    return SumoLoadConfig(rd1, rd2, rd3, veh_flow, general, ev_routes, evs, load_args, ev_powers)
