import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
import torch
from torch_geometric.data import Data
from css.experiment.config import EMA_feature
from css.experiment import planning,training
from css.graph import features
from css.sumo import traffic_signals as signals


class DynamicMetricSelectionTests(unittest.TestCase):
    def test_all_feature_modes_match_unconditional_collector_over_time(self):
        modes={'all':(0,1,2),'dynamic_only':(0,1,2),'static_only':(),
               'travel_time_only':(0,),'avg_speed_only':(1,),'occupancy_only':(2,),
               'static_travel_time':(0,),'static_occupancy':(2,),'static_travel_time_occupancy':(0,2)}
        clock=[0.]
        getters=[Mock(side_effect=lambda edge:20.+clock[0]),
                 Mock(side_effect=lambda edge:10.+clock[0]/10),
                 Mock(side_effect=lambda edge:.1+clock[0]/100)]
        fake=SimpleNamespace(simulation=SimpleNamespace(getTime=lambda:clock[0]),
            edge=SimpleNamespace(getTraveltime=getters[0],getLastStepMeanSpeed=getters[1],getLastStepOccupancy=getters[2]),
            TraCIException=RuntimeError)
        static={'data_static':Data(x=torch.tensor([[1.,2.,3.],[4.,5.,6.]]),edge_index=torch.tensor([[0,1],[1,0]])),
                'edge_ids':['a','b'],'eid2eidx':{'a':0,'b':1},'static_edge_attr_norm':torch.full((2,5),.5)}
        def legacy(edge_ids,tt,ms,occ,debug=False,**kw):
            now=fake.simulation.getTime()
            return torch.tensor([[tt.update(e,getters[0](e),now),ms.update(e,getters[1](e),now),
                                  occ.update(e,getters[2](e),now)] for e in edge_ids],dtype=torch.float32)
        with patch.object(features,'traci',fake):
            for debug in [False,True]:
                for mode,required in modes.items():
                    with self.subTest(mode=mode,debug=debug):
                        old_emas=[EMA_feature(half_life_sec=5) for _ in range(3)]
                        expected=[]
                        with patch.object(features,'collect_dynamic_edge_traveltime',side_effect=legacy),contextlib.redirect_stdout(io.StringIO()):
                            for now in [0.,.5,3.]:
                                clock[0]=now;expected.append(features.build_data_runtime(static,*old_emas,debug=debug,edge_feature_mode=mode))
                        for getter in getters:getter.reset_mock()
                        new_emas=[EMA_feature(half_life_sec=5) for _ in range(3)]
                        with contextlib.redirect_stdout(io.StringIO()):
                            for now,reference in zip([0.,.5,3.],expected):
                                clock[0]=now;actual=features.build_data_runtime(static,*new_emas,debug=debug,edge_feature_mode=mode)
                                for field in ['x','edge_index','edge_attr']:
                                    torch.testing.assert_close(getattr(actual,field),getattr(reference,field),rtol=0,atol=0)
                        selected=(0,1,2) if debug else required
                        self.assertEqual([g.call_count for g in getters],[6 if i in selected else 0 for i in range(3)])
                        for i in selected:
                            self.assertEqual(new_emas[i].val,old_emas[i].val)
                            self.assertEqual(new_emas[i].last_t,old_emas[i].last_t)

    def test_collector_default_and_empty_edge_shape_remain_compatible(self):
        fake=SimpleNamespace(simulation=SimpleNamespace(getTime=lambda:0),edge=SimpleNamespace(
            getTraveltime=Mock(return_value=5),getLastStepMeanSpeed=Mock(return_value=6),getLastStepOccupancy=Mock(return_value=.2)),TraCIException=RuntimeError)
        with patch.object(features,'traci',fake):
            emas=[EMA_feature(half_life_sec=5) for _ in range(3)]
            torch.testing.assert_close(features.collect_dynamic_edge_traveltime(['a'],*emas),torch.tensor([[5.,6.,.2]]))
            self.assertEqual(features.collect_dynamic_edge_traveltime([],*emas).shape,(0,3))
            with self.assertRaises(ValueError):features.collect_dynamic_edge_traveltime(['a'],*emas,metric_indices=(9,))


class SignalTopologyCacheTests(unittest.TestCase):
    def setUp(self):signals.reset_signal_topology_cache()
    def tearDown(self):signals.reset_signal_topology_cache()

    def fake(self):
        return SimpleNamespace(trafficlight=SimpleNamespace(getIDList=Mock(return_value=['tls']),
            getControlledLinks=Mock(return_value=[[("in_0","out_0","via")]])))

    def test_topology_is_reused_and_reset_discards_previous_scenario(self):
        fake=self.fake()
        with patch.object(signals,'traci',fake),patch.object(planning,'traci',fake):
            for _ in range(3):
                self.assertEqual(signals.find_tls_and_link_index_for_movement('in','out'),('tls',0))
                self.assertEqual(planning.resolve_movement_out_lanes('in','out'),['out_0'])
            self.assertEqual(fake.trafficlight.getIDList.call_count,1)
            self.assertEqual(fake.trafficlight.getControlledLinks.call_count,1)
            fake.trafficlight.getIDList.return_value=['new_tls']
            fake.trafficlight.getControlledLinks.return_value=[[("in_0","other_0","via")]]
            signals.reset_signal_topology_cache()
            self.assertEqual(signals.find_tls_and_link_index_for_movement('in','out'),(None,None))
            self.assertEqual(signals.find_tls_and_link_index_for_movement('in','other'),('new_tls',0))
            self.assertEqual(fake.trafficlight.getIDList.call_count,2)

    def test_changing_traci_domain_does_not_reuse_another_connection(self):
        first=self.fake();second=self.fake();second.trafficlight.getIDList.return_value=['second']
        self.assertEqual(signals.get_tls_ids(traci_api=first),('tls',))
        self.assertEqual(signals.get_tls_ids(traci_api=second),('second',))

    def test_phase_time_and_program_queries_remain_live(self):
        fake=self.fake();clock=[10.];phase=[0];switch=[30.]
        fake.simulation=SimpleNamespace(getTime=lambda:clock[0])
        fake.trafficlight.getPhase=Mock(side_effect=lambda tls:phase[0])
        fake.trafficlight.getNextSwitch=Mock(side_effect=lambda tls:switch[0])
        fake.trafficlight.getProgram=Mock(return_value='p')
        fake.trafficlight.getAllProgramLogics=Mock(return_value=[SimpleNamespace(programID='p',getPhases=lambda:[SimpleNamespace(state='r',duration=30.),SimpleNamespace(state='G',duration=20.)])])
        with patch.object(signals,'traci',fake):
            signals.get_tls_ids();signals.get_controlled_links('tls')
            self.assertEqual(signals.time_to_green_and_remaining_green('tls',0),(20.,20.))
            clock[0]=31.;phase[0]=1;switch[0]=50.
            self.assertEqual(signals.time_to_green_and_remaining_green('tls',0),(0.,19.))
            self.assertEqual(fake.trafficlight.getAllProgramLogics.call_count,2)
            self.assertEqual(fake.trafficlight.getNextSwitch.call_count,2)

    def test_evaluation_load_explicitly_invalidates_topology(self):
        fake=self.fake();fake.load=Mock();fake.simulation=SimpleNamespace(getMinExpectedNumber=lambda:0,getTime=lambda:0.)
        env=SimpleNamespace(episode_reset=lambda *a:None,nid2idx={},set_route_planning_agent=lambda *a:None)
        cfg=SimpleNamespace(load_args=[],evs=[],ev_power={})
        with patch.object(training,'traci',fake),patch.object(signals,'traci',fake):
            self.assertEqual(signals.get_tls_ids(),('tls',))
            fake.trafficlight.getIDList.return_value=['next']
            training._run_evaluation_rollout(env,None,cfg,'dijkstra')
            self.assertEqual(signals.get_tls_ids(),('next',))

if __name__=='__main__':unittest.main()
