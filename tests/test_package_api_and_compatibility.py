from __future__ import annotations

import unittest
from pathlib import Path

import config
import env
import training
import utils_sumo
from DQN.base_dqn import base_DQN as legacy_base_dqn
from SAC.sacd import DiscreteSACAgent as legacy_sacd
from contextual_bandit import ContextualLinUCBAgent as legacy_linucb
from paper_baseline import PaperBaselineAgent as legacy_paper
from route_planning_agent.DQN.route_dqn import route_DQN as legacy_route_dqn

from css import experiment, graph, routing, sumo
from css.agents.charging import (
    ContextualLinUCBAgent,
    DiscreteSACAgent,
    PaperBaselineAgent,
    base_DQN,
)
from css.agents.charging.dqn import DEFAULT_DQN_CHECKPOINT
from css.agents.charging.paper_baseline import DEFAULT_PAPER_BASELINE_CHECKPOINT
from css.agents.charging.sacd import DEFAULT_SACD_CHECKPOINT
from css.agents.routing import route_DQN
from css.charging import Env


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PublicPackageApiTests(unittest.TestCase):
    def test_domain_packages_publish_expected_operations(self):
        self.assertTrue({"dijkstra_path", "cost_by_length"}.issubset(routing.__all__))
        self.assertTrue(
            {
                "apply_full_route",
                "count_evs_in_rou",
                "remaining_routes_len",
                "time_to_green_and_remaining_green",
            }.issubset(sumo.__all__)
        )
        self.assertTrue({"build_data_runtime", "prepare_static"}.issubset(graph.__all__))
        self.assertTrue({"ScenarioKey", "DRL_train", "evaluation"}.issubset(experiment.__all__))

    def test_private_helpers_are_not_public_package_api(self):
        self.assertNotIn("_safe_get_lane", sumo.__all__)
        self.assertNotIn("_pick_active_logic", sumo.__all__)
        self.assertNotIn("tls_node_features", graph.__all__)


class CompatibilityTests(unittest.TestCase):
    def test_root_module_shims_preserve_object_identity(self):
        self.assertIs(config.EVState, experiment.EVState)
        self.assertIs(env.Env, Env)
        self.assertIs(training.ScenarioKey, experiment.ScenarioKey)
        self.assertIs(training.DRL_train, experiment.DRL_train)
        self.assertIs(utils_sumo.dijkstra_path, routing.dijkstra_path)
        self.assertIs(utils_sumo.apply_full_route, sumo.apply_full_route)
        self.assertIs(utils_sumo.prepare_static, graph.prepare_static)

    def test_agent_shims_preserve_object_identity(self):
        self.assertIs(legacy_linucb, ContextualLinUCBAgent)
        self.assertIs(legacy_sacd, DiscreteSACAgent)
        self.assertIs(legacy_paper, PaperBaselineAgent)
        self.assertIs(legacy_base_dqn, base_DQN)
        self.assertIs(legacy_route_dqn, route_DQN)

    def test_default_checkpoint_paths_still_point_to_project_parameters(self):
        self.assertEqual(DEFAULT_DQN_CHECKPOINT, PROJECT_ROOT / "parameters" / "env_dqn.pt")
        self.assertEqual(DEFAULT_SACD_CHECKPOINT, PROJECT_ROOT / "parameters" / "env_sacd.pt")
        self.assertEqual(
            DEFAULT_PAPER_BASELINE_CHECKPOINT,
            PROJECT_ROOT / "parameters" / "env_paper_baseline.pt",
        )

    def test_facade_describes_package_replacements(self):
        self.assertIs(utils_sumo.DEPRECATED_FACADE, True)
        self.assertEqual(utils_sumo.FACADE_REPLACEMENTS["sumo"], "css.sumo")

    def test_undefined_tls_feature_schema_fails_explicitly(self):
        with self.assertRaisesRegex(NotImplementedError, "no defined feature schema"):
            utils_sumo.tls_node_features("tls-0")


if __name__ == "__main__":
    unittest.main()
