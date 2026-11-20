"""Parallel charging, reset behavior, and validation split/capacity plumbing."""
import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from css.charging import Env
from css.experiment.environments import resolve_environment
from css.experiment import training
from scripts import run_experiment, run_validation


class ChargerCapacityTests(unittest.TestCase):
    def make_env(self, capacity):
        env = Env.__new__(Env)
        env.cs_loc = ['a', 'b', 'c']
        env.chargers_per_station = capacity
        env.cs_init()
        return env

    def test_four_simultaneous_arrivals_charge_without_waiting_then_fifth_waits(self):
        for capacity in (3, 4):
            env = self.make_env(capacity)
            waits = [env.update(100., 'a', .2)[0] for _ in range(5)]
            self.assertEqual(waits[:capacity], [0.] * capacity)
            self.assertEqual(waits[capacity], 900.)
            self.assertEqual(env.usage['b'], [0.] * capacity)
            self.assertEqual(env.charge(.2), 900.)
            self.assertIsNot(env.usage['a'], env.usage['b'])

    def test_reset_keeps_capacity_and_clears_queues(self):
        env = self.make_env(4)
        env.update(10., 'a', .2)
        env.reset()
        self.assertEqual(env.cs_pile, 4)
        self.assertEqual(env.usage['a'], [0.] * 4)
        self.assertEqual(env.eat['a'], 0)

    def test_default_capacity_and_invalid_constructor_values(self):
        env = Env.__new__(Env)
        env.cs_loc = ['a']
        env.cs_init()
        self.assertEqual(env.usage['a'], [0.] * 3)
        for invalid in (0, -1, 2.5, True, '4'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'positive integer'):
                Env([], {}, None, chargers_per_station=invalid)
        args = run_experiment.parse_args([])
        self.assertEqual(args.chargers_per_station, 3)
        self.assertEqual(args.eval_split, 'testing')
        for invalid in ('0', '-1', '2.5'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_experiment.parse_args(['--chargers-per-station', invalid])

    def test_validation_rejects_training_or_higher_dataset_before_sumo(self):
        for argv in (['--eval-split','validation'],
                     ['--eval-split','validation','--mode','eval','--higher-evaluation']):
            with self.assertRaisesRegex(ValueError, 'requires --mode eval'):
                run_experiment.main(argv)

    def test_all_methods_both_environments_use_requested_split_capacity_and_separate_output(self):
        for env_name in ('env','new_env'):
            for agent_name, route in [('linucb','dqngnn'),('sacd','dqngnn'),('edqn','dqn'),('paper_baseline','dijkstra')]:
                for capacity in (3,4):
                    with self.subTest(env=env_name, agent=agent_name, capacity=capacity), tempfile.TemporaryDirectory() as directory:
                        root=Path(directory)
                        checkpoint=root/'charging.pt'
                        checkpoint.write_bytes(b'fixed checkpoint')
                        selected=replace(resolve_environment(env_name), root=root/env_name)
                        agent=Mock()
                        agent.network=torch.nn.Sequential(torch.nn.Linear(1,1),torch.nn.Dropout(.5))
                        detail={'evaluation_scope': {'split':'validation'}, 'methods': {}}
                        result=(0.,0.,0.,(0.,0.,0.),(0.,0.,0.),(0.,0.,0.),detail)
                        with contextlib.ExitStack() as stack:
                            for name,value in [('PROJECT_ROOT',root), ('resolve_environment',selected),
                                               ('validate_environment_assets',{}),('require_route_model',None),
                                               ('checkBinary','sumo'),('readNet',Mock()),('prepare_static',{}),
                                               ('Env',Mock(state_dim=7,action_dim=3)),
                                               ('ContextualLinUCBAgent',agent),('DiscreteSACAgent',agent),
                                               ('base_eDQN',agent),('PaperBaselineAgent',agent)]:
                                if name=='PROJECT_ROOT':
                                    stack.enter_context(patch.object(run_experiment,name,value))
                                else:
                                    stack.enter_context(patch.object(run_experiment,name,return_value=value))
                            stack.enter_context(patch.object(run_experiment,'traci'))
                            stack.enter_context(patch('torch.utils.tensorboard.SummaryWriter'))
                            evaluator=stack.enter_context(patch.object(run_experiment,'evaluation',return_value=result))
                            train=stack.enter_context(patch.object(run_experiment,'DRL_train'))
                            saver=stack.enter_context(patch.object(run_experiment,'save_training_results'))
                            constructor=run_experiment.Env
                            argv=['--env',env_name,'--agent',agent_name,'--route-mode',route,
                                  '--mode','eval','--eval-split','validation','--no-eval-references',
                                  '--chargers-per-station',str(capacity),'--checkpoint',str(checkpoint)]
                            with contextlib.redirect_stdout(io.StringIO()):
                                self.assertEqual(run_experiment.main(argv),0)
                            train.assert_not_called()
                            self.assertEqual(constructor.call_args.kwargs['chargers_per_station'],capacity)
                            self.assertEqual(evaluator.call_args.kwargs['split'],'validation')
                            self.assertFalse(agent.network.training)
                            saved,output=saver.call_args.args
                            self.assertEqual(output.parent,root/'outputs'/env_name/f'validation_cs{capacity}')
                            self.assertTrue(output.name.startswith(f'validation_detail_{agent_name}_{route}_'))
                            self.assertEqual(saved['charging_infrastructure']['total_chargers'],3*capacity)
                            self.assertEqual(saved['validation_provenance']['chargers_per_station'],capacity)

    def test_real_validation_selects_full_manifest_and_no_testing_cases(self):
        for env_name in ('env','new_env'):
            selected=resolve_environment(env_name)
            seen=[]
            def rollout(env,agent,cfg,mode,rollout_mode):
                seen.append((cfg.rd3,cfg.rd2,cfg.rd1))
                return {'total_reward':-3.,'breakdown':(-1.,-2.,0.),
                        'travel_cost_minutes':1.,'waiting_cost_minutes':2.,
                        'route_failure_count':0,'route_failures':[]}
            with patch.object(training,'_run_evaluation_rollout',side_effect=rollout), \
                 patch.object(training,'tqdm',side_effect=lambda x,**kw:x):
                result=training.evaluation(Mock(),Mock(),split='validation',environment=selected,return_details=True)
            self.assertEqual(len(seen),90)
            self.assertEqual(len(set(seen)),90)
            self.assertEqual(result[-1]['evaluation_scope']['split'],'validation')
            self.assertEqual(result[-1]['evaluation_scope']['general_data_dir'],'validation_data')

    def test_batch_validation_never_trains_and_propagates_capacity_to_all_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'scripts').mkdir()
            (root/'scripts/run_experiment.py').touch()
            for route,checkpoint in run_validation.jobs_for(root,'new_env').values():
                checkpoint.parent.mkdir(parents=True,exist_ok=True)
                checkpoint.touch()
                if route!='dijkstra':
                    name='dqngnn_best_better_ratio.pt' if route=='dqngnn' else 'dqn_best_better_ratio.pt'
                    model=root/'route_planning_agent/DQN/new_env'/name
                    model.parent.mkdir(parents=True,exist_ok=True)
                    model.touch()
            with patch.object(run_validation.subprocess,'run') as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_validation.main(['--project-dir',str(root),'--env','new_env',
                                                      '--chargers-per-station','4']),0)
            self.assertEqual(run.call_count,4)
            for call in run.call_args_list:
                cmd=call.args[0]
                self.assertEqual(cmd[cmd.index('--mode')+1],'eval')
                self.assertEqual(cmd[cmd.index('--eval-split')+1],'validation')
                self.assertEqual(cmd[cmd.index('--chargers-per-station')+1],'4')
                self.assertIn('--no-eval-references',cmd)
                self.assertEqual(call.kwargs['env']['PYTHONHASHSEED'],'42')


if __name__=='__main__':
    unittest.main()
