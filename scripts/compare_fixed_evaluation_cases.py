from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


IGNORED_KEYS = {"elapsed_seconds", "label", "project", "checkpoint"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare fixed CSS evaluation case files.")
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--refactor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-9)
    return parser.parse_args(argv)


def compare_values(original, refactor, path, differences, tolerance):
    if isinstance(original, dict) and isinstance(refactor, dict):
        keys = (set(original) | set(refactor)) - IGNORED_KEYS
        for key in sorted(keys):
            if key not in original or key not in refactor:
                differences.append(
                    {"path": f"{path}.{key}", "original": original.get(key), "refactor": refactor.get(key)}
                )
                continue
            compare_values(
                original[key], refactor[key], f"{path}.{key}", differences, tolerance
            )
        return

    if isinstance(original, list) and isinstance(refactor, list):
        if len(original) != len(refactor):
            differences.append(
                {"path": f"{path}.length", "original": len(original), "refactor": len(refactor)}
            )
        for index, (old_item, new_item) in enumerate(zip(original, refactor)):
            compare_values(
                old_item, new_item, f"{path}[{index}]", differences, tolerance
            )
        return

    if (
        isinstance(original, (int, float))
        and not isinstance(original, bool)
        and isinstance(refactor, (int, float))
        and not isinstance(refactor, bool)
    ):
        if not math.isclose(float(original), float(refactor), rel_tol=0.0, abs_tol=tolerance):
            differences.append({"path": path, "original": original, "refactor": refactor})
        return

    if original != refactor:
        differences.append({"path": path, "original": original, "refactor": refactor})


def summarize(payload):
    records = payload.get("cases", payload.get("episodes", []))
    ev_states = [state for record in records for state in record["evs"].values()]
    goals = Counter(state["goal_cs"] for state in ev_states)
    summary = {
        "record_type": "cases" if "cases" in payload else "episodes",
        "record_count": len(records),
        "ev_count": sum(record["ev_count"] for record in records),
        "completed_ev_count": sum(bool(state["done"]) for state in ev_states),
        "out_ev_count": sum(bool(state["out"]) for state in ev_states),
        "goal_counts": dict(sorted(goals.items())),
    }
    if "cases" in payload:
        summary.update(
            {
                "travel_cost_minutes": sum(
                    case["aggregate"]["travel_cost_minutes"] for case in records
                ),
                "waiting_cost_minutes": sum(
                    case["aggregate"]["waiting_cost_minutes"] for case in records
                ),
                "travel_waiting_total_minutes": sum(
                    case["aggregate"]["travel_waiting_total_minutes"] for case in records
                ),
            }
        )
    else:
        summary.update(
            {
                "reward_sum": sum(payload["train_rewards"]),
                "final_action_counts": payload["final_agent"]["action_counts"],
                "final_learn_step_counter": payload["final_agent"]["learn_step_counter"],
                "final_memory_size": payload["final_agent"]["memory_size"],
            }
        )
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    original = json.loads(args.original.read_text(encoding="utf-8"))
    refactor = json.loads(args.refactor.read_text(encoding="utf-8"))
    differences = []
    compare_values(original, refactor, "root", differences, args.absolute_tolerance)
    report = {
        "equivalent": not differences,
        "absolute_tolerance": args.absolute_tolerance,
        "difference_count": len(differences),
        "differences": differences,
        "original_summary": summarize(original),
        "refactor_summary": summarize(refactor),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if differences:
        print(f"fixed evaluation differs in {len(differences)} fields; see {args.output}")
        return 1
    print(f"fixed evaluation results are equivalent; report saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
