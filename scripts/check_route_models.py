"""Check bundled route models locally; does not start SUMO or train."""
import json
from pathlib import Path
import torch
from sumolib.net import readNet
from css.agents.routing.bundles import validate_route_bundle
from css.agents.routing.dqn import route_DQN,DQNShapeConfig
from css.agents.routing.dqngnn import route_DQNGNN,DQNGNNActionShapeConfig
from css.graph import get_edge_index_v2


def main():
    root=Path(__file__).resolve().parents[1]
    rows=[]
    for name in ['env','new_env']:
        _,_,mapping,_,_=get_edge_index_v2(readNet(str(root/name/'env_static_tls.net.xml'),withInternal=True))
        folder=root/'route_planning_agent/DQN'/name
        for mode in ['dqngnn','dqn']:
            path=folder/f'{mode}_best_better_ratio.pt'
            validate_route_bundle(path,name,mapping)
            c=torch.load(path,map_location='cpu',weights_only=True)
            if mode=='dqn':agent=route_DQN(DQNShapeConfig(len(mapping),c['shape']['max_neighbors']),device=torch.device('cpu'),memory_capacity=4)
            else:agent=route_DQNGNN(DQNGNNActionShapeConfig(**c['shape']),device=torch.device('cpu'),memory_capacity=4)
            agent.load_full_model(filename=str(path),load_optimizer=False)
            rows.append(dict(environment=name,route_mode=mode,episode_one_based=c['episode']+1,
                input_schema=c['shape'],status='passed'))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
