from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


def build_snapshot(project: Path) -> dict:
    sys.path.insert(0, str(project))

    import config
    import training
    import utils_sumo
    from contextual_bandit import ContextualLinUCBAgent
    from env import Env

    schedule = [
        asdict(key)
        for key in training.build_training_scenario_schedule(8, schedule_seed=42)
    ]

    ema = config.EMA_feature(half_life_sec=10.0)
    ema_values = [
        ema.update("edge", 10.0, 100.0),
        ema.update("edge", 20.0, 110.0),
        ema.update("edge", 30.0, 110.0),
    ]

    environment = Env.__new__(Env)
    environment.cs_loc = ["cs-a", "cs-b", "cs-c"]
    environment.dis_table = {"origin": [0.1, 0.2, 0.3]}
    environment.eat = {"cs-a": 100.0, "cs-b": 700.0, "cs-c": 1_300.0}
    environment.usage = {
        "cs-a": [100.0, 200.0, 300.0],
        "cs-b": [700.0, 800.0, 900.0],
        "cs-c": [1_300.0, 1_400.0, 1_500.0],
    }
    environment.avai_port = {"cs-a": 0, "cs-b": 0, "cs-c": 0}
    state = environment.generate_sacd_state(700.0, "origin", 0.4)
    waiting, charging = environment.update(50.0, "cs-a", 0.2)

    route_file = project / "env" / "training_data" / "ev_1800_p01_1.rou.xml"
    route_ids = utils_sumo.get_evs_in_rou(str(route_file))

    agent = ContextualLinUCBAgent(
        2,
        2,
        alpha=0.0,
        quadratic_features=False,
        enable_tensorboard=False,
        seed=7,
    )
    for _ in range(20):
        agent.memory.push([0.25, 0.75], [0], [1.0], None, True)
        agent.memory.push([0.25, 0.75], [1], [-1.0], None, True)
    agent.update_parameters()

    return {
        "schedule": schedule,
        "ema": ema_values,
        "charging_state": state,
        "queue_update": {
            "waiting": waiting,
            "charging": charging,
            "usage": environment.usage["cs-a"],
            "eat": environment.eat["cs-a"],
            "available_port": int(environment.avai_port["cs-a"]),
        },
        "route_ids": route_ids,
        "linucb_action": agent.choose_action([0.25, 0.75], eval=True),
        "linucb_counts": agent.action_counts.tolist(),
    }


def snapshot_in_subprocess(python: Path, project: Path) -> dict:
    command = [
        str(python),
        str(Path(__file__).resolve()),
        "--snapshot-project",
        str(project),
    ]
    completed = subprocess.run(
        command,
        cwd=project,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Snapshot failed for {project}:\n{completed.stdout}\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare deterministic CSS behavior snapshots.")
    parser.add_argument("--old", type=Path)
    parser.add_argument("--new", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-difference",
        action="append",
        default=[],
        help="Top-level snapshot field intentionally changed by a validated fix.",
    )
    parser.add_argument("--snapshot-project", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.snapshot_project is not None:
        print(json.dumps(build_snapshot(args.snapshot_project.resolve()), sort_keys=True))
        return 0
    if args.old is None or args.new is None:
        raise SystemExit("--old and --new are required")

    old_snapshot = snapshot_in_subprocess(args.python, args.old.resolve())
    new_snapshot = snapshot_in_subprocess(args.python, args.new.resolve())
    different_keys = sorted(
        key
        for key in set(old_snapshot) | set(new_snapshot)
        if old_snapshot.get(key) != new_snapshot.get(key)
    )
    allowed_differences = sorted(set(args.allow_difference))
    unexpected_differences = sorted(set(different_keys) - set(allowed_differences))
    result = {
        "equal": old_snapshot == new_snapshot,
        "different_keys": different_keys,
        "allowed_differences": allowed_differences,
        "unexpected_differences": unexpected_differences,
        "old": old_snapshot,
        "new": new_snapshot,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if result["equal"]:
        print("CSS behavior snapshots match.")
        return 0
    if not unexpected_differences:
        print(
            "CSS behavior snapshots differ only in allowed fields: "
            + ", ".join(different_keys)
        )
        return 0
    print("Unexpected CSS behavior differences: " + ", ".join(unexpected_differences))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
