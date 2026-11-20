"""Evaluate the four existing CSS checkpoints on validation without training."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("linucb", "sacd", "edqn", "paper_baseline")


def jobs_for(project, environment):
    root = project / "outputs" / ("all_new_env" if environment == "new_env" else "all_old_env")
    if environment == "new_env":
        return {
            "linucb": ("dqngnn", root / "linucb_route_dqngnn_final.pt"),
            "sacd": ("dqngnn", root / "sacd_route_dqngnn.pt"),
            "edqn": ("dqn", root / "dqn_route_dqn.pt"),
            "paper_baseline": ("dijkstra", root / "paper_baseline_route_dijkstra.pt"),
        }
    return {
        "linucb": ("dqngnn", root / "linucb_route_dqngnn_701515_final.pt"),
        "sacd": ("dqngnn", root / "sacd_route_dqngnn_701515_final.pt"),
        "edqn": ("dqn", project / "parameters/env/edqn_route_dqn_exploration_v2_final.pt"),
        "paper_baseline": ("dijkstra", root / "paper_baseline_route_dijkstra_701515_final.pt"),
    }


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("env", "new_env"), default="new_env")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--agent", choices=METHODS, help="Omit to run all four methods sequentially.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chargers-per-station", type=positive_int, default=3)
    parser.add_argument("--check-only", action="store_true", help="Check assets without starting SUMO.")
    return parser.parse_args(argv)


def native_arguments(args, agent, route, checkpoint):
    return ["--env", args.env, "--agent", agent, "--route-mode", route,
            "--mode", "eval", "--eval-split", "validation", "--seed", str(args.seed),
            "--chargers-per-station", str(args.chargers_per_station),
            "--checkpoint", str(checkpoint), "--no-eval-references"]


def main(argv=None):
    args = parse_args(argv)
    project = args.project_dir.expanduser().resolve()
    if not (project / "scripts/run_experiment.py").is_file():
        raise FileNotFoundError("CSS project not found: " + str(project))
    jobs = jobs_for(project, args.env)
    selected = [args.agent] if args.agent else list(METHODS)
    # Check every requested model before starting the first long evaluation.
    for agent in selected:
        route, checkpoint = jobs[agent]
        if not checkpoint.is_file():
            raise FileNotFoundError("Checkpoint not found: " + str(checkpoint))
        if route != "dijkstra":
            name = "dqngnn_best_better_ratio.pt" if route == "dqngnn" else "dqn_best_better_ratio.pt"
            route_model = project / "route_planning_agent/DQN" / args.env / name
            if not route_model.is_file():
                raise FileNotFoundError("Route checkpoint not found: " + str(route_model))
    child_env = os.environ.copy()
    child_env.update(PYTHONHASHSEED=str(args.seed), PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    sumo_home = Path(sys.prefix) / "Lib/site-packages/sumo"
    if sumo_home.is_dir():
        child_env["SUMO_HOME"] = str(sumo_home)
    for agent in selected:
        route, checkpoint = jobs[agent]
        native = native_arguments(args, agent, route, checkpoint)
        if args.check_only:
            if agent != selected[0]:
                print("[OK]", agent, "+", route, checkpoint)
                continue
            native.append("--check-data")
        print("Checking inputs:" if args.check_only else "Starting validation:",
              agent, "+", route, "| chargers per station:", args.chargers_per_station, flush=True)
        subprocess.run([sys.executable, "-B", "-m", "scripts.run_experiment", *native],
                       cwd=project, env=child_env, check=True)
    print("Inputs checked; SUMO was not started." if args.check_only else "All requested validation runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
