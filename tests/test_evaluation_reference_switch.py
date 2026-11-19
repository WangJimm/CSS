"""Final-testing references are optional without changing the selected policy."""
import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from css.experiment import training
from css.experiment.environments import resolve_environment
from scripts import run_experiment


class EvaluationReferenceSwitchTests(unittest.TestCase):
    def test_references_remain_enabled_by_default(self):
        self.assertFalse(run_experiment.parse_args([]).no_eval_references)
        self.assertTrue(run_experiment.parse_args(['--no-eval-references']).no_eval_references)

    def test_cli_wiring_for_both_environments_all_four_methods_and_both_evaluators(self):
        methods = [('linucb', 'dqngnn'), ('sacd', 'dqngnn'),
                   ('edqn', 'dqn'), ('paper_baseline', 'dijkstra')]
        for env_name in ('env', 'new_env'):
            for agent_name, route in methods:
                for higher in (False, True):
                    for references in (False, True):
                        with self.subTest(env=env_name, agent=agent_name, higher=higher,
                                          references=references), tempfile.TemporaryDirectory() as directory:
                            root = Path(directory)
                            checkpoint = root / 'charging.pt'
                            checkpoint.touch()
                            selected = replace(resolve_environment(env_name), root=root / env_name)
                            agent = Mock()
                            output = io.StringIO()
                            detail = {'methods': {'agent': {'total_cost': 123.},
                                                  'min_waiting': {} if references else None,
                                                  'shortest': {} if references else None}}
                            result = (1., 0., 0., (1., 2., 3.), (0., 0., 0.), (0., 0., 0.), detail)
                            with contextlib.ExitStack() as stack:
                                for name, value in [('resolve_environment', selected),
                                                    ('validate_environment_assets', {}),
                                                    ('require_route_model', None),
                                                    ('require_higher_evaluation_assets', None),
                                                    ('checkBinary', 'sumo.exe'),
                                                    ('readNet', Mock()), ('prepare_static', {}),
                                                    ('Env', Mock(state_dim=7, action_dim=3)),
                                                    ('ContextualLinUCBAgent', agent),
                                                    ('DiscreteSACAgent', agent),
                                                    ('base_eDQN', agent),
                                                    ('PaperBaselineAgent', agent)]:
                                    stack.enter_context(patch.object(run_experiment, name, return_value=value))
                                stack.enter_context(patch.object(run_experiment, 'traci'))
                                stack.enter_context(patch('torch.utils.tensorboard.SummaryWriter'))
                                standard = stack.enter_context(patch.object(run_experiment, 'evaluation', return_value=result))
                                extended = stack.enter_context(patch.object(run_experiment, 'higher_evaluation', return_value=result))
                                train = stack.enter_context(patch.object(run_experiment, 'DRL_train'))
                                save = stack.enter_context(patch.object(run_experiment, 'save_training_results'))
                                argv = ['--env', env_name, '--agent', agent_name, '--route-mode', route,
                                        '--mode', 'eval', '--checkpoint', str(checkpoint)]
                                if higher:
                                    argv.append('--higher-evaluation')
                                if not references:
                                    argv.append('--no-eval-references')
                                with contextlib.redirect_stdout(output):
                                    self.assertEqual(run_experiment.main(argv), 0)
                            called, unused = (extended, standard) if higher else (standard, extended)
                            called.assert_called_once()
                            unused.assert_not_called()
                            train.assert_not_called()
                            self.assertEqual(called.call_args.kwargs['using_min_waiting'], references)
                            self.assertEqual(called.call_args.kwargs['using_shortest'], references)
                            self.assertEqual(called.call_args.kwargs['route_planning_mode'], route)
                            agent.load_para.assert_called_once_with(str(checkpoint))
                            saved = save.call_args.args[0]
                            self.assertEqual(saved['evaluation_options'], {
                                'using_min_waiting': references, 'using_shortest': references})
                            self.assertEqual(saved['methods']['agent']['total_cost'], 123.)
                            self.assertEqual('reference strategies skipped' in output.getvalue(), not references)
                            self.assertEqual('min waiting (0.00' in output.getvalue(), references)

    def test_real_evaluator_keeps_all_agent_cases_and_marks_skipped_methods_null(self):
        # Real manifest/data loading and aggregation; only SUMO rollout is replaced.
        for env_name in ('env', 'new_env'):
            selected = resolve_environment(env_name)
            runs = {}
            for references in (True, False):
                seen = []
                def rollout(env, agent, cfg, mode, rollout_mode):
                    seen.append((cfg.rd3, cfg.rd2, cfg.rd1, rollout_mode))
                    return {'total_reward': -3., 'breakdown': (-1., -2., 0.),
                            'travel_cost_minutes': 1., 'waiting_cost_minutes': 2.,
                            'route_failure_count': 0, 'route_failures': []}
                with patch.object(training, '_run_evaluation_rollout', side_effect=rollout), \
                     patch.object(training, 'tqdm', side_effect=lambda x, **kw: x):
                    result = training.evaluation(Mock(), Mock(), using_min_waiting=references,
                                                 using_shortest=references, environment=selected,
                                                 return_details=True)
                self.assertEqual(len(seen), 270 if references else 90)
                agent_cases = [case[:3] for case in seen if case[3] == 'agent']
                self.assertEqual(len(set(agent_cases)), 90)
                runs[references] = (result, agent_cases)
                if not references:
                    self.assertIsNone(result[-1]['methods']['min_waiting'])
                    self.assertIsNone(result[-1]['methods']['shortest'])
            self.assertEqual(runs[True][1], runs[False][1])
            self.assertEqual(runs[True][0][0], runs[False][0][0])
            self.assertEqual(runs[True][0][3], runs[False][0][3])
            self.assertEqual(runs[True][0][-1]['methods']['agent'],
                             runs[False][0][-1]['methods']['agent'])


if __name__ == '__main__':
    unittest.main()
