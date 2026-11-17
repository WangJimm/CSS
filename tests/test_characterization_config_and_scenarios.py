from __future__ import annotations

import json
import tempfile
import unittest
import random
from pathlib import Path

import numpy as np

import config
import training
from css.experiment import scenarios


class ConfigCharacterizationTests(unittest.TestCase):
    def test_checkout_paths_and_flow_groups_are_stable(self):
        project_root = Path(__file__).resolve().parents[1]

        self.assertEqual(config.HERE, project_root)
        self.assertEqual(config.RUN_DIR, project_root / "env")
        self.assertEqual(config.training_veh_flows, [1800])
        self.assertEqual(config.validation_veh_flows, [1800])
        self.assertEqual(config.testing_veh_flows, [1500, 1800, 2100])
        self.assertEqual((config.TRAINING_SEED_START, config.TRAINING_SEED_END), (1, 20))
        self.assertEqual(config.ROUTE_PLANNING_MODE, ["dqngnn", "dqn", "dijkstra", "edqn"])

    def test_ema_uses_elapsed_time_and_half_life(self):
        ema = config.EMA_feature(half_life_sec=10.0)

        self.assertEqual(ema.update("edge", 10.0, 100.0), 10.0)
        self.assertAlmostEqual(ema.update("edge", 20.0, 110.0), 15.0)
        self.assertAlmostEqual(ema.update("edge", 30.0, 110.0), 15.0)


class ScenarioCharacterizationTests(unittest.TestCase):
    def test_seeded_training_schedule_matches_current_sequence(self):
        self.assertEqual(
            training.build_training_scenario_schedule(8, schedule_seed=42),
            [
                training.ScenarioKey(0, 1, 9),
                training.ScenarioKey(0, 1, 5),
                training.ScenarioKey(0, 1, 19),
                training.ScenarioKey(0, 1, 1),
                training.ScenarioKey(0, 1, 8),
                training.ScenarioKey(0, 1, 18),
                training.ScenarioKey(0, 1, 15),
                training.ScenarioKey(0, 1, 6),
            ],
        )

    def test_every_default_training_scenario_has_required_assets(self):
        for seed in range(config.TRAINING_SEED_START, config.TRAINING_SEED_END + 1):
            for probability in (1, 2):
                loaded = training.load_sumo(
                    scenario_key=training.ScenarioKey(0, probability, seed),
                    veh_flows=config.training_veh_flows,
                )
                with self.subTest(seed=seed, probability=probability):
                    self.assertTrue(Path(loaded.general).is_file())
                    self.assertTrue(Path(loaded.ev_routes).is_file())
                    self.assertEqual(len(loaded.evs), len(loaded.ev_power))

    def test_flow_index_resolution_and_error_are_stable(self):
        flows = [1500, 1800, 2100]

        self.assertEqual(training._default_flow_indices(flows), [0, 1, 2])
        self.assertEqual(training._resolve_veh_flow(flows, 2), 2100)
        with self.assertRaisesRegex(ValueError, "flow_index 3"):
            training._resolve_veh_flow(flows, 3)

    def test_json_result_writer_converts_numpy_scalars_atomically(self):
        payload = {"score": np.float32(1.25), "nested": (np.int64(2),)}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "result.json"
            training.save_training_results(payload, output)
            loaded = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(loaded, {"score": 1.25, "nested": [2]})
            self.assertEqual(list(output.parent.glob("*.tmp")), [])

    def test_training_loader_maps_real_route_ids_to_power_values(self):
        loaded = training.load_sumo(
            scenario_key=training.ScenarioKey(0, 1, 1),
            veh_flows=[1800],
        )

        self.assertEqual((loaded.rd1, loaded.rd2, loaded.rd3), (1, 1, 0))
        self.assertEqual(loaded.veh_flow, 1800)
        self.assertEqual(Path(loaded.general).name, "output_1800_1.rou.xml")
        self.assertEqual(Path(loaded.general).parent.name, "training_data")
        self.assertEqual(Path(loaded.ev_routes).name, "ev_1800_p01_1.rou.xml")
        self.assertEqual(Path(loaded.ev_routes).parent.name, "training_data")
        self.assertEqual(list(loaded.ev_power), loaded.evs)
        self.assertEqual(len(loaded.evs), len(loaded.ev_power))
        self.assertIn("--route-files", loaded.load_args)

    def test_explicit_scenario_does_not_consume_global_random_state(self):
        random.seed(1234)
        before = random.getstate()

        training.load_sumo(
            scenario_key=training.ScenarioKey(0, 1, 1),
            veh_flows=[1800],
        )

        self.assertEqual(random.getstate(), before)

    def test_higher_evaluation_fails_before_loading_when_dataset_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "higher_test_data"):
                scenarios.require_higher_evaluation_assets(Path(directory))

    def test_loader_rejects_ev_power_cardinality_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            routes = root / "ev_case.rou.xml"
            powers = root / "power_case.json"
            routes.write_text(
                "<routes><vehicle id='EV_0'/><vehicle id='EV_1'/></routes>",
                encoding="utf-8",
            )
            powers.write_text("[0.2]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "2 EVs but 1 power values"):
                scenarios._load_ev_power_mapping(routes, powers)


if __name__ == "__main__":
    unittest.main()
