"""Prevent old-env split leakage and accidental changes to RP inputs."""
from __future__ import annotations

import contextlib
import io
import json
import random
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from css.experiment import scenarios, training
from css.experiment.environments import resolve_environment
from scripts import run_experiment


class OldEnvironmentSplitTests(unittest.TestCase):
    def test_both_environments_have_exact_stratification_and_disjoint_seeds(self):
        for name in ('env', 'new_env'):
            env = resolve_environment(name)
            self.assertTrue(env.manifest_based)
            seeds = {}
            for split in ('training', 'validation', 'testing'):
                keys = scenarios.scenario_keys_for_split(env, split)
                expected = 70 if split == 'training' else 15
                self.assertEqual(Counter((k.scenario_index, k.probability_index) for k in keys),
                                 {(i, p): expected for i in range(3) for p in (1, 2)})
                seeds[split] = {k.seed for k in keys}
                groups = Counter((k.scenario_index, k.seed) for k in keys)
                self.assertTrue(all(n == 2 for n in groups.values()))
            for a, b in (('training', 'validation'), ('training', 'testing'), ('validation', 'testing')):
                self.assertFalse(seeds[a] & seeds[b])

    def test_default_old_schedule_uses_full_training_pool_and_is_reproducible(self):
        env = resolve_environment()
        before = random.getstate()
        a = scenarios.build_training_scenario_schedule(1000, schedule_seed=42)
        self.assertEqual(a, scenarios.build_training_scenario_schedule(1000, schedule_seed=42))
        self.assertEqual(before, random.getstate())
        self.assertTrue(set(a) <= set(scenarios.scenario_keys_for_split(env, 'training')))
        self.assertEqual({k.scenario_index for k in a}, {0, 1, 2})
        self.assertEqual({k.probability_index for k in a}, {1, 2})
        self.assertGreater(len(set(a)), 350)

    def test_old_loader_rejects_testing_seed_in_training(self):
        env = resolve_environment()
        key = scenarios.scenario_keys_for_split(env, 'testing')[0]
        with self.assertRaisesRegex(ValueError, 'not in env/training'):
            scenarios.load_sumo(scenario_key=key)

    def test_old_data_check_inspects_all_ev_power_and_goals_without_sumo(self):
        with patch.object(run_experiment.traci, 'start') as start, \
             patch.object(run_experiment, 'require_route_model') as model, \
             contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(run_experiment.main(['--env', 'env', '--check-data']), 0)
        start.assert_not_called()
        model.assert_not_called()
        summary = json.loads(stream.getvalue())
        self.assertEqual(summary['scenario_counts'], {'training': 420, 'validation': 90, 'testing': 90})
        self.assertEqual(summary['ev_power_assignments'], 8200)

    def test_old_validation_and_testing_read_separate_ninety_case_pools(self):
        env = resolve_environment()
        seen = {}
        def rollout(_env, _agent, cfg, mode, rollout_mode):
            seen.setdefault(cfg.split, set()).add((cfg.veh_flow, cfg.rd2, cfg.rd1))
            self.assertEqual(Path(cfg.general).parent.name, f'{cfg.split}_data')
            return {'total_reward': -1., 'breakdown': (1., 0., 0.), 'travel_cost_minutes': 1.,
                    'waiting_cost_minutes': 0., 'route_failure_count': 0, 'route_failures': []}
        with patch.object(training, '_run_evaluation_rollout', side_effect=rollout), \
             patch.object(training, 'tqdm', side_effect=lambda x, **kw: x):
            for split in ('validation', 'testing'):
                result = training.evaluation(Mock(), Mock(), environment=env, split=split, return_details=True)
                self.assertEqual(result[-1]['evaluation_scope']['scenario_count'], 90)
        self.assertEqual(len(seen['validation']), 90)
        self.assertEqual(len(seen['testing']), 90)
        self.assertFalse(seen['validation'] & seen['testing'])

    def test_default_training_validates_on_validation_and_preserves_station_action_order(self):
        env_context = resolve_environment()
        self.assertEqual(env_context.stations, ('42496394', '42490303', '42474000'))
        cfg = SimpleNamespace(load_args=[], evs=[], ev_power={}, rd1=30000, rd2=1, rd3=0,
                              veh_flow=1500, ev_routes='ev.xml', general='general.xml')
        env = Mock(); env.nid2idx = {'node': 0}
        agent = Mock(); agent.memory = []; agent.batch_size = 1
        result = (0., 0., 0., (0., 0., 0.), (0., 0., 0.), (0., 0., 0.), {})
        with tempfile.TemporaryDirectory() as directory, patch.object(training, 'load_sumo', return_value=cfg), \
             patch.object(training, 'traci') as traci, patch.object(training, 'evaluation', return_value=result) as validate, \
             patch.object(training, 'tqdm', side_effect=lambda x, **kw: x), \
             patch.object(training, '_results_output_dir', return_value=Path(directory)):
            traci.simulation.getMinExpectedNumber.return_value = 0
            traci.simulation.getTime.return_value = 0.
            training.DRL_train(env, agent, [], episodes=1, save_results=False,
                               checkpoint_filename=Path(directory) / 'mock.pt')
        self.assertEqual(validate.call_args.kwargs['split'], 'validation')
        self.assertEqual(validate.call_args.kwargs['veh_flows'], [1500, 1800, 2100])

    def test_manifest_does_not_change_old_route_signal_features(self):
        old = resolve_environment()
        new = resolve_environment('new_env')
        self.assertTrue(old.manifest_based and new.manifest_based)
        self.assertFalse(old.prefer_controlled_tls)
        self.assertTrue(new.prefer_controlled_tls)
        with tempfile.TemporaryDirectory() as directory:
            selected = replace(old, root=Path(directory) / 'env')
            fake = Mock(); fake.state_dim = 7; fake.action_dim = 3
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(run_experiment, 'resolve_environment', return_value=selected))
                stack.enter_context(patch.object(run_experiment, 'validate_environment_assets', return_value={}))
                stack.enter_context(patch.object(run_experiment, 'checkBinary', return_value='sumo.exe'))
                stack.enter_context(patch.object(run_experiment, 'traci'))
                stack.enter_context(patch.object(run_experiment, 'readNet'))
                prepare = stack.enter_context(patch.object(run_experiment, 'prepare_static'))
                stack.enter_context(patch.object(run_experiment, 'Env', return_value=fake))
                stack.enter_context(patch.object(run_experiment, 'DiscreteSACAgent'))
                stack.enter_context(patch.object(run_experiment, 'CompactReplayBuffer'))
                stack.enter_context(patch.object(run_experiment, 'plot_train_process'))
                train = stack.enter_context(patch.object(run_experiment, 'DRL_train', return_value=([], [], [], [], None, None, None)))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run_experiment.main(['--env', 'env', '--route-mode', 'dijkstra', '--episodes', '1']), 0)
            prepare.assert_called_once_with(unittest.mock.ANY, prefer_controlled_tls=False)
            self.assertTrue(train.call_args.kwargs['environment'].manifest_based)


if __name__ == '__main__':
    unittest.main()
