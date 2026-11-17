from __future__ import annotations

import subprocess
import sys
import unittest
import ast
from pathlib import Path

from scripts import compare_fixed_evaluation_cases, run_experiment
from css.charging import require_route_model, route_model_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScriptEntrypointTests(unittest.TestCase):
    def test_import_has_no_simulation_or_output_side_effect(self):
        self.assertTrue(callable(run_experiment.main))
        self.assertTrue(callable(run_experiment.parse_args))

    def test_parser_keeps_existing_defaults(self):
        args = run_experiment.parse_args([])

        self.assertEqual(args.agent, "sacd")
        self.assertEqual(args.mode, "train")
        self.assertEqual(args.episodes, 1000)
        self.assertEqual(args.seed, 42)
        self.assertFalse(args.higher_evaluation)

    def test_higher_evaluation_is_explicit_opt_in(self):
        args = run_experiment.parse_args(["--higher-evaluation"])

        self.assertTrue(args.higher_evaluation)
        self.assertFalse(args.standard_evaluation)

    def test_route_checkpoint_is_separate_from_charging_checkpoint(self):
        args = run_experiment.parse_args(
            [
                "--route-checkpoint",
                "parameters/route.pt",
                "--checkpoint",
                "parameters/charging.pt",
            ]
        )

        self.assertEqual(args.route_checkpoint, Path("parameters/route.pt"))
        self.assertEqual(args.checkpoint, "parameters/charging.pt")

    def test_all_route_checkpoints_exist(self):
        self.assertTrue(route_model_path("dqn").is_file())
        self.assertTrue(route_model_path("dqngnn").is_file())
        self.assertTrue(route_model_path("edqn").is_file())
        self.assertEqual(require_route_model("edqn"), route_model_path("edqn"))

    def test_missing_higher_dataset_fails_before_sumo_start(self):
        with self.assertRaisesRegex(FileNotFoundError, "higher_test_data"):
            run_experiment.main(["--higher-evaluation"])

    def test_main_contains_no_success_path_sys_exit_calls(self):
        tree = ast.parse(
            (PROJECT_ROOT / "scripts" / "run_experiment.py").read_text(encoding="utf-8")
        )
        sys_exit_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sys"
            and node.func.attr == "exit"
        ]
        self.assertEqual(sys_exit_calls, [])

    def test_module_and_compatibility_help_commands_succeed(self):
        commands = (
            [sys.executable, "-m", "scripts.run_experiment", "--help"],
            [sys.executable, "cs_navigation.py", "--help"],
        )
        for command in commands:
            with self.subTest(command=command):
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("Train or evaluate CS-selection agents", completed.stdout)

    def test_fixed_result_comparator_summarizes_training_payloads(self):
        payload = {
            "episodes": [
                {
                    "ev_count": 1,
                    "evs": {
                        "EV_0": {
                            "done": True,
                            "out": False,
                            "goal_cs": "station",
                        }
                    },
                }
            ],
            "train_rewards": [-1.5],
            "final_agent": {
                "action_counts": [1],
                "learn_step_counter": 1,
                "memory_size": 1,
            },
        }

        summary = compare_fixed_evaluation_cases.summarize(payload)

        self.assertEqual(summary["record_type"], "episodes")
        self.assertEqual(summary["completed_ev_count"], 1)
        self.assertEqual(summary["reward_sum"], -1.5)


if __name__ == "__main__":
    unittest.main()
