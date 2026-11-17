from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ROOTS = {
    "config",
    "contextual_bandit",
    "DQN",
    "env",
    "paper_baseline",
    "route_planning_agent",
    "SAC",
    "training",
    "utils_sumo",
}
ALLOWED_LEGACY_IMPORTERS = {
    Path("scripts/compare_version_snapshot.py"),
    Path("scripts/run_fixed_evaluation_cases.py"),
    Path("scripts/run_fixed_training_cases.py"),
}


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_package_and_scripts_do_not_import_legacy_roots(self):
        violations = []
        for source_root in (PROJECT_ROOT / "css", PROJECT_ROOT / "scripts"):
            for path in source_root.rglob("*.py"):
                if path.relative_to(PROJECT_ROOT) in ALLOWED_LEGACY_IMPORTERS:
                    continue
                forbidden = imported_roots(path) & FORBIDDEN_ROOTS
                if forbidden:
                    violations.append(f"{path.relative_to(PROJECT_ROOT)}: {sorted(forbidden)}")
        self.assertEqual(violations, [])

    def test_root_python_files_are_shims_except_utils_facade(self):
        for path in PROJECT_ROOT.glob("*.py"):
            if path.name == "utils_sumo.py":
                continue
            with self.subTest(path=path.name):
                first_line = path.read_text(encoding="utf-8").splitlines()[0]
                self.assertIn("Compatibility", first_line)

    def test_utils_facade_contains_no_function_or_class_implementation(self):
        tree = ast.parse((PROJECT_ROOT / "utils_sumo.py").read_text(encoding="utf-8"))
        implementations = [
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        ]
        self.assertEqual(implementations, [])

    def test_training_uses_environment_route_model_validation(self):
        source = (PROJECT_ROOT / "css" / "experiment" / "training.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("env.set_route_planning_agent(route_planning_mode)", source)
        self.assertNotIn("env.route_planning_agent =", source)

    def test_reusable_package_code_never_terminates_the_process(self):
        violations = []
        for path in (PROJECT_ROOT / "css").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "sys"
                    and node.func.attr == "exit"
                ):
                    violations.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
