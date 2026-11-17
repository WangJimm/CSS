from __future__ import annotations
import contextlib
import io
import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from sumolib.net import readNet
from css.experiment.environments import resolve_environment
from css.experiment import scenarios, training
from css.charging.environment import Env, route_model_path
from css.agents.routing.dqngnn import route_DQNGNN
from css.graph import features
from scripts import run_experiment

class EnvironmentSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = resolve_environment('env')
        cls.new = resolve_environment('new_env')

    def test_cli_selects_environment_without_changing_default(self):
        self.assertEqual(run_experiment.parse_args([]).env, 'env')
        self.assertEqual(run_experiment.parse_args(['--env', 'new_env']).env, 'new_env')
        self.assertTrue(run_experiment.parse_args(['--env','new_env','--check-data']).check_data)

    def test_context_switch_does_not_mutate_legacy_globals(self):
        from css.experiment import config
        self.assertEqual(config.RUN_DIR.name, 'env')
        self.assertEqual(config.training_veh_flows, [1800])
        self.assertEqual(self.new.training_flows, (2650,2800,2950))
        self.assertEqual(self.new.stations, ('1419061109','446290091','5984003368'))
        self.assertEqual(resolve_environment().root, self.old.root)

    def test_manifest_has_exact_counts_and_no_seed_leakage(self):
        pools = {split: scenarios.scenario_keys_for_split(self.new, split) for split in ('training','validation','testing')}
        self.assertEqual({k:len(v) for k,v in pools.items()}, {'training':420,'validation':90,'testing':90})
        self.assertFalse({k.seed for k in pools['training']} & {k.seed for k in pools['validation']})
        self.assertFalse({k.seed for k in pools['validation']} & {k.seed for k in pools['testing']})
        self.assertFalse({k.seed for k in pools['training']} & {k.seed for k in pools['testing']})

    def test_manifest_training_schedule_reproducible_and_training_only(self):
        before = random.getstate()
        a = scenarios.build_training_scenario_schedule(1000, schedule_seed=42, environment=self.new)
        b = scenarios.build_training_scenario_schedule(1000, schedule_seed=42, environment=self.new)
        self.assertEqual(a,b)
        self.assertEqual(before,random.getstate())
        self.assertTrue(set(a).issubset(set(scenarios.scenario_keys_for_split(self.new,'training'))))
        self.assertNotEqual(a,scenarios.build_training_scenario_schedule(1000,schedule_seed=43,environment=self.new))

    def test_loader_uses_real_ids_power_and_selected_config_for_all_600_cases(self):
        total = 0
        for split in ('training','validation','testing'):
            for key in scenarios.scenario_keys_for_split(self.new,split):
                cfg = scenarios.load_sumo(scenario_key=key, environment=self.new, split=split)
                self.assertEqual(Path(cfg.ev_routes).parent,self.new.root/f'{split}_data')
                self.assertEqual(list(cfg.ev_power),cfg.evs)
                self.assertTrue(all(.2 <= v <= .4 for v in cfg.ev_power.values()))
                self.assertEqual(cfg.load_args[1],str(self.new.cfg_path))
                self.assertEqual(cfg.split,split)
                total += len(cfg.evs)
        self.assertEqual(total,12700)

    def test_loader_rejects_test_seed_in_training(self):
        key = scenarios.scenario_keys_for_split(self.new,'testing')[0]
        with self.assertRaisesRegex(ValueError,'not in new_env/training'):
            scenarios.load_sumo(scenario_key=key,environment=self.new,split='training')

    def test_filters_select_only_existing_scenarios(self):
        keys = scenarios.scenario_keys_for_split(self.new,'validation',flow_indices=[1],probability_indices=[2])
        self.assertEqual(len(keys),15)
        self.assertTrue(all(k.scenario_index == 1 and k.probability_index == 2 for k in keys))

    def test_check_data_never_starts_sumo_or_requires_model(self):
        with patch.object(run_experiment.traci,'start') as start, patch.object(run_experiment,'require_route_model') as model:
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(run_experiment.main(['--env','new_env','--check-data']),0)
            summary = json.loads(stream.getvalue())
        start.assert_not_called();model.assert_not_called()
        self.assertEqual(summary['scenario_counts'],{'training':420,'validation':90,'testing':90})
        self.assertEqual(summary['ev_power_assignments'],12700)

    def test_missing_new_route_model_fails_before_sumo(self):
        with patch.object(run_experiment.traci,'start') as start, \
             patch.object(run_experiment,'require_route_model',side_effect=FileNotFoundError('new_env route model missing')):
            with self.assertRaisesRegex(FileNotFoundError,'new_env'):
                run_experiment.main(['--env','new_env'])
        start.assert_not_called()

    def test_model_and_output_paths_do_not_fall_back_to_old_environment(self):
        self.assertEqual(route_model_path('dqngnn','new_env').parent.name,'new_env')
        self.assertNotEqual(route_model_path('dqngnn','new_env'),route_model_path('dqngnn','env'))
        self.assertEqual(self.new.output_dir.name,'new_env')
        self.assertEqual(self.new.model_dir.name,'new_env')
        self.assertEqual(self.old.output_dir.name,'outputs')
        self.assertIsNone(route_model_path('dijkstra','new_env'))

    def test_dijkstra_does_not_load_any_unrelated_route_checkpoint(self):
        env = Env.__new__(Env)
        env.environment_name='new_env'
        env.route_planning_agent_dqngnn=Mock();env.route_planning_agent_dqn=Mock();env.route_planning_agent_edqn=Mock()
        env.load_route_planning_agents(modes=['dijkstra'])
        self.assertEqual(env.available_route_modes,{'dijkstra'})
        env.route_planning_agent_dqngnn.load_full_model.assert_not_called()
        env.route_planning_agent_dqn.load_full_model.assert_not_called()
        env.route_planning_agent_edqn.load_full_model.assert_not_called()

    def test_explicit_route_model_loads_only_selected_mode(self):
        env = Env.__new__(Env);env.environment_name='new_env'
        env.route_planning_agent_dqngnn=Mock();env.route_planning_agent_dqn=Mock();env.route_planning_agent_edqn=Mock()
        path=route_model_path('dqngnn','env')
        env.load_route_planning_agents(dqngnn_filename=path,modes=['dqngnn'])
        env.route_planning_agent_dqngnn.load_full_model.assert_called_once_with(filename=str(path),buffer_name=None)
        env.route_planning_agent_dqn.load_full_model.assert_not_called()
        env.route_planning_agent_edqn.load_full_model.assert_not_called()
        self.assertEqual(env.available_route_modes,{'dijkstra','dqngnn'})

    def test_route_shape_mismatch_rejected_even_if_network_weights_could_fit(self):
        agent=route_DQNGNN.__new__(route_DQNGNN)
        agent.device=torch.device('cpu');agent.shape=SimpleNamespace(num_nodes=379)
        with patch('css.agents.routing.dqngnn.torch.load',return_value={'shape':SimpleNamespace(num_nodes=115)}) as load:
            with self.assertRaisesRegex(ValueError,'num_nodes=115, expected 379'):
                agent.load_full_model(filename='old.pt')
        self.assertFalse(load.call_args.kwargs['weights_only'])

    def test_validation_and_testing_use_distinct_complete_case_sets(self):
        seen={}
        def rollout(env,agent,cfg,mode,rollout_mode):
            seen.setdefault(cfg.split,[]).append((cfg.rd3,cfg.rd2,cfg.rd1))
            return {'total_reward':-1.,'breakdown':(1.,0.,0.),'travel_cost_minutes':1.,'waiting_cost_minutes':0.,'route_failure_count':0,'route_failures':[]}
        with patch.object(training,'_run_evaluation_rollout',side_effect=rollout),patch.object(training,'tqdm',side_effect=lambda x,**kw:x):
            for split in ('validation','testing'):
                result=training.evaluation(Mock(),Mock(),environment=self.new,split=split,return_details=True)
                self.assertEqual(result[0],-90.)
                self.assertEqual(result[-1]['evaluation_scope']['general_data_dir'],f'{split}_data')
                self.assertEqual(result[-1]['evaluation_scope']['scenario_count'],90)
        self.assertEqual(len(set(seen['validation'])),90)
        self.assertEqual(len(set(seen['testing'])),90)
        self.assertFalse(set(seen['validation']) & set(seen['testing']))

    def test_training_wires_validation_to_validation_split(self):
        fake_cfg=SimpleNamespace(load_args=[],evs=[],ev_power={},rd1=50001,rd2=1,rd3=0,veh_flow=2650,ev_routes='ev.xml',general='general.xml')
        env=Mock();env.nid2idx={'node':0}
        agent=Mock();agent.memory=[];agent.batch_size=1
        evaluation=(0.,0.,0.,(0.,0.,0.),(0.,0.,0.),(0.,0.,0.),{})
        with tempfile.TemporaryDirectory() as directory, patch.object(training,'load_sumo',return_value=fake_cfg), \
             patch.object(training,'traci') as traci, patch.object(training,'evaluation',return_value=evaluation) as validate, \
             patch.object(training,'_results_output_dir',return_value=Path(directory)), \
             patch.object(training,'tqdm',side_effect=lambda x,**kw:x):
            traci.simulation.getMinExpectedNumber.return_value=0
            traci.simulation.getTime.return_value=0.
            training.DRL_train(env,agent,[],episodes=1,environment=self.new,save_results=False,
                               checkpoint_filename=Path(directory)/'mock.pt')
        self.assertEqual(validate.call_args.kwargs['split'],'validation')
        self.assertIs(validate.call_args.kwargs['environment'],self.new)

    def test_cli_training_passes_selected_context_and_isolated_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            selected = replace(self.new, root=Path(directory)/"new_env")
            fake_env = Mock(); fake_env.state_dim=7; fake_env.action_dim=3
            agent = Mock()
            results=([],[],[],[],None,None,None)
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(run_experiment,'resolve_environment',return_value=selected))
                stack.enter_context(patch.object(run_experiment,'validate_environment_assets',return_value={}))
                stack.enter_context(patch.object(run_experiment,'checkBinary',return_value='sumo.exe'))
                stack.enter_context(patch.object(run_experiment,'traci'))
                stack.enter_context(patch.object(run_experiment,'readNet'))
                prepare=stack.enter_context(patch.object(run_experiment,'prepare_static'))
                constructor=stack.enter_context(patch.object(run_experiment,'Env',return_value=fake_env))
                stack.enter_context(patch.object(run_experiment,'DiscreteSACAgent',return_value=agent))
                stack.enter_context(patch.object(run_experiment,'CompactReplayBuffer',return_value=[]))
                train=stack.enter_context(patch.object(run_experiment,'DRL_train',return_value=results))
                with contextlib.redirect_stdout(io.StringIO()):
                    code=run_experiment.main(['--env','new_env','--route-mode','dijkstra','--episodes','3'])
            self.assertEqual(code,0)
            self.assertIs(train.call_args.kwargs['environment'],selected)
            self.assertEqual(Path(train.call_args.kwargs['checkpoint_filename']),selected.model_dir/'sacd_route_dijkstra.pt')
            self.assertEqual(constructor.call_args.kwargs['environment_name'],'new_env')
            self.assertEqual(constructor.call_args.args[0],list(self.new.stations))
            prepare.assert_called_once_with(unittest.mock.ANY,prefer_controlled_tls=True)

    def test_new_environment_uses_real_tls_controller_ids(self):
        net=readNet(str(self.new.net_path),withInternal=True)
        first=features.get_edge_index_v2(net)[1][0]
        count=len(features.get_edge_index_v2(net)[1])
        with patch.object(features.traci.trafficlight,'getIDList',return_value=['GS_controller']), \
             patch.object(features,'build_inedge_to_tls_and_indices',return_value={first:{'GS_controller':[0]}}), \
             patch.object(features,'compute_signal_backbone_for_edges',return_value=([0.]*count,[1.]*count)) as backbone:
            features.prepare_static(net,prefer_controlled_tls=True)
            self.assertEqual(backbone.call_args.args[1][0],'GS_controller')
            features.prepare_static(net)
            self.assertIsNone(backbone.call_args.args[1][0])

if __name__ == '__main__': unittest.main()
