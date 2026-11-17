from pathlib import Path
from dataclasses import asdict
import json
import tempfile
import unittest
import torch
from sumolib.net import readNet
from css.agents.routing.dqn import DQNShapeConfig,route_DQN
from css.agents.routing.bundles import validate_route_bundle
from css.graph import get_edge_index_v2

ROOT=Path(__file__).resolve().parents[1]

class ImportedRouteModelTests(unittest.TestCase):
    def test_shipped_tensor_bundles_match_graph_and_load_dqn_time_schema(self):
        for env,n in [('env',115),('new_env',379)]:
            _,_,mapping,_,_=get_edge_index_v2(readNet(str(ROOT/env/'env_static_tls.net.xml'),withInternal=True))
            folder=ROOT/'route_planning_agent/DQN'/env
            for mode in ['dqngnn','dqn']:
                path=folder/f'{mode}_best_better_ratio.pt'
                validate_route_bundle(path,env,mapping)
                checkpoint=torch.load(path,map_location='cpu',weights_only=True)
                self.assertEqual(checkpoint['css_inference_format'],'rp_tensors_v1')
                self.assertEqual(checkpoint['shape']['num_nodes'],n)
            agent=route_DQN(DQNShapeConfig(n,4),device=torch.device('cpu'),memory_capacity=4)
            agent.load_full_model(filename=str(folder/'dqn_best_better_ratio.pt'))
            self.assertEqual(agent.shape.dqn_input_dim,2*n+9)
            base=torch.zeros(2*n);base[0]=1;base[n+1]=1
            observation=agent.build_shared_observation(base,[[1.,2.],[3.,4.]],1800)
            torch.testing.assert_close(observation[-9:],torch.tensor([1.,2.,3.,4.,0.,0.,0.,0.,.5]))
            self.assertTrue(torch.isfinite(agent.eval_net(observation)).all())

    def test_legacy_tls_only_checkpoint_remains_supported(self):
        agent=route_DQN(DQNShapeConfig(3,2),device=torch.device('cpu'),memory_capacity=4)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'legacy.pt'
            torch.save({'shape':asdict(agent.shape),'eval_q_net':agent.eval_net.state_dict(),'target_q_net':agent.target_net.state_dict()},path)
            agent.load_full_model(filename=str(path))
            self.assertEqual(agent.build_shared_observation(torch.zeros(6),[[1,2]],123).numel(),10)

    def test_wrong_network_rejected_before_loading_state_dict(self):
        agent=route_DQN(DQNShapeConfig(115,4),device=torch.device('cpu'),memory_capacity=4)
        path=ROOT/'route_planning_agent/DQN/new_env/dqn_best_better_ratio.pt'
        with self.assertRaisesRegex(ValueError,'num_nodes=379'):
            agent.load_full_model(filename=str(path))
        with self.assertRaisesRegex(ValueError,'environment'):
            validate_route_bundle(path,'env',{})

if __name__=='__main__':unittest.main()
