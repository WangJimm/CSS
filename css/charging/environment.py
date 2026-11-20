"""
建立充電站使用的環境
"""
import sys
import numpy as np
from pathlib import Path
from css.agents.routing.dqngnn import DQNGNNActionShapeConfig, route_DQNGNN
from css.agents.routing.dqn import DQNShapeConfig, route_DQN
from css.agents.routing.edqn import EDQNShapeConfig, route_eDQN
from css.agents.routing.bundles import validate_route_bundle
import heapq as hq

from css.routing import cost_by_length, dijkstra_path
from css.sumo import count_evs_in_rou


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTE_MODEL_DIR = PROJECT_ROOT / "route_planning_agent" / "DQN" / "env"
ROUTE_MODEL_FILENAMES = {
    "dqngnn": "dqngnn_best_better_ratio.pt",
    "dqn": "dqn_best_better_ratio.pt",
    "edqn": "edqn_best_better_ratio.pt",
}


def route_model_path(mode: str, environment_name: str = "env") -> Path | None:
    if mode == "dijkstra":
        return None
    try:
        if environment_name not in ("env", "new_env"):
            raise ValueError(f"Unknown environment: {environment_name}")
        directory = ROUTE_MODEL_DIR if environment_name == "env" else PROJECT_ROOT / "route_planning_agent" / "DQN" / environment_name
        return directory / ROUTE_MODEL_FILENAMES[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown route planning mode: {mode}") from exc


def require_route_model(mode: str, environment_name: str = "env") -> Path | None:
    path = route_model_path(mode, environment_name)
    if path is not None and not path.is_file():
        raise FileNotFoundError(
            f"Route-planning mode {mode!r} for {environment_name!r} requires a checkpoint that is not installed: {path}. "
            "Use --route-checkpoint with a model trained for this network, or --route-mode dijkstra."
        )
    return path


def normalize_state_minmax(state, state_min, state_max):
    return (state - state_min) / (state_max - state_min)


class Env(object):
    def __init__(self, cs_loc, sumo_static_pack, net, using_gnn = True, *, environment_name="env", route_mode=None, route_checkpoint=None, chargers_per_station=3):
        if isinstance(chargers_per_station, bool) or not isinstance(chargers_per_station, (int, np.integer)) or chargers_per_station < 1:
            raise ValueError("chargers_per_station must be a positive integer")
        self.chargers_per_station = int(chargers_per_station)
        self.environment_name = environment_name
        self.action_dim = len(cs_loc) ## action = 有幾個CS要選
        self.state_dim = 2*len(cs_loc) + 1   
        # 目前 dis:len(CS), Soc:1, CS_state:len(CS), 第幾個 1, not useing: avai_piles:3
        

        self.max_power = 52
        self.cs_loc = cs_loc

        self.sumo_static_pack = sumo_static_pack
        self.net = net
        # self.edge_ids = sumo_static_pack["edge_ids"] # 每個edge在sumo裡面的id
        self.nid2idx = sumo_static_pack["nid2idx"] # type是dict  keys為sumo的node_id  vals為從0開始的編號
        self.eid2eidx = sumo_static_pack['eid2eidx'] # type是dict  keys為sumo的edge_id  vals為從0開始的編號
        self.adj = sumo_static_pack["adj"]
        self.node_in = sumo_static_pack["data_static"].x.size(1) 
        self.edge_in = 3 + sumo_static_pack["static_edge_attr_norm"].size(1)

        self.max_neighbors = max(len(v) for v in self.adj.values())

        dqngnn_shape = DQNGNNActionShapeConfig(
            num_nodes=len(self.nid2idx),
            max_neighbors=self.max_neighbors,
            node_in=self.node_in + 4,
            edge_in=self.edge_in,
            action_feature_dim=self.edge_in + 7,
        )
        dqn_shape = DQNShapeConfig(
            num_nodes=len(self.nid2idx),
            max_neighbors=self.max_neighbors,
        )

        edqn_shape = EDQNShapeConfig(
            num_nodes=len(self.nid2idx),
            max_neighbors=self.max_neighbors,
        )

        self.route_planning_agent_dqngnn = route_DQNGNN(dqngnn_shape)
        self.route_planning_agent_dqn = route_DQN(dqn_shape)
        self.route_planning_agent_edqn = route_eDQN(edqn_shape)
        if route_mode is None:
            self.load_route_planning_agents()
            self.set_route_planning_agent("dqngnn" if using_gnn else "dqn")
        else:
            overrides = {f"{route_mode}_filename": route_checkpoint} if route_checkpoint is not None else {}
            self.load_route_planning_agents(modes=[route_mode], **overrides)
            self.set_route_planning_agent(route_mode)

        self.build_distance_table()
        self.cs_init()
        # self.ev_init() # 一開始沒有load ev時不知道EV數量沒辦法生成
        self.achieve = 0 # 確認抵達狀況

        ### 不同目標成本
        self.waiting_time = 0
        self.charging_time = 0
        self.driving_time = 0



    def build_distance_table(self):
        self.dis_table = {} # dict
        max_d = 0
        min_d = 0
        for node in self.nid2idx:
            # if node in self.cs_loc:
            #     continue
            ## 根據dijkstra算距離
            dis = []
            for cs in self.cs_loc:
                if node != cs:
                    path_nodes, path_edges, total = dijkstra_path(net = self.net, adj = self.adj, cur_edge = None, cur_node = node, goal_node=cs, cost_fn=cost_by_length)
                    if total:
                        max_d = max(max_d, total)
                    else: # 有些node沒辦法抵達, 隨便給個值，因為也不會出現在狀態中
                        total = 1
                else:
                    total = 0.0
                
                dis.append(total)
            self.dis_table[node] = dis

        ########################
        ## 正規化
        ########################
        for node in self.nid2idx:
            for i in range(len(self.dis_table[node])):
                self.dis_table[node][i] = normalize_state_minmax(self.dis_table[node][i], min_d, max_d)

    def cs_init(self):
        """
        建立充電站的一些參數
        1.充電樁數量
        2.每一個充電樁的使用狀況(剩幾秒才能充完電)
        3.這幾個充電站的最早可充電時間
        4.這幾個充電站的充電花費
        """
        #充電樁數量
        self.cs_pile = getattr(self, "chargers_per_station", 3)
        # self.usage = np.zeros((len(self.cs_loc),self.cs_pile)) #使用狀況 shape [充電站數量, 每個充電站的充電樁數量]
        # self.eat = np.zeros(len(self.cs_loc)) # 最早可充電時間 shape [充電站數量]
        # self.avai_port = np.zeros(len(self.cs_loc)) # pointer, 指向目前空閒的充電樁數量
        self.charging_cost = [] # 設定每個充電站的充電花費 (錢的部分)
        self.cs_Queue = {}
        # self.end_waiting_times = [[]  for _ in range(len(self.cs_loc))] #每個CS的排隊中EV的結束排隊時間, shape = [len(cs_loc)]
        

        self.usage = {}
        self.eat = {}
        self.avai_port = {}
        self.end_waiting_times = {}
        for i, cs in enumerate(self.cs_loc):
            self.charging_cost.append(i+7)
            self.cs_Queue[cs] = []
            self.usage[cs] = [0.0] * self.cs_pile
            self.eat[cs] = 0
            self.avai_port[cs] = 0
            self.end_waiting_times[cs] = []

    def ev_init(self):
        """
        建立電動車的相關狀態與參數
        # 1.先決定這一天有幾台ev {num}
        # 2.根據這個num 平均的以選定時間為mean做normal distribution
        # 3.幫每一個ev隨機搭配一個位置{G的節點}
        ############################################################## 前面三點在sumo都已經決定好了，後續電量也要看sumo決定的數量再處理
        # 4.加上電量
        """
        
        """ 生成電量 """
        pass
        # self.ev_power = np.round(np.random.uniform(0.2,0.4,self.ev_num),2)

        

    def load_ev(self, ev_routes):
        """
        每個episode開始時要先從ev flow讀取有幾台EV

        """
        self.ev_num = count_evs_in_rou(ev_routes)
        
    def episode_reset(self, evs, ev_powers):
        """ 
        每個episode要重製一次 
        
        evs : list, 裡面裝ev編號如 EV_0, EV_1, ...
        """
        self.cs_init()
        self.ev_init()
        self.waiting_time = 0
        self.charging_time = 0
        self.driving_time = 0  
        self.evs = evs
        self.ev_powers = ev_powers




    def reset(self):
        ## 重製充電站
        self.cs_init()
        self.ev_init()
        self.waiting_time = 0
        self.charging_time = 0
        self.driving_time = 0  

    def generate_sacd_state(self, depart_time, depart_node, depart_power):
        """
        state[1] = 現在時間
        更換成 depart_time

        state[0] = 現在位置(node)
        更換成 depart_node

        state[2] = 電量
        更換成 depart_power

        [(state[1]-dt)/86400] = 到目前過了多少時間，在這個環境應該用不到

        return 
        sacd_state = [dis_to_cs] + [ev_power] + [earliest available time(eat)]
        
        
        """
        # EAT is stored as an absolute SUMO timestamp.  Agent state should
        # describe how long the EV must wait relative to its own departure,
        # matching the contextual LinUCB state semantics.  Stations that are
        # already available have zero remaining wait.
        eat_state = [
            max(float(self.eat[cs]) - float(depart_time), 0.0) / 600.0
            for cs in self.cs_loc
        ]
        dis_state = self.dis_table[depart_node]
        sacd_state = dis_state + [depart_power] + eat_state # + [(state[1]-dt)/86400] # + [state[0]/env.node_len] # + avai_piles #DRL_state 是要丟給 DRL的
        return sacd_state
        

    def set_route_planning_agent(self, mode = "dqngnn"):
        if mode == "dijkstra":
            return
        if mode not in self.available_route_modes:
            path = self.route_model_paths.get(mode, route_model_path(mode, getattr(self, "environment_name", "env")))
            raise FileNotFoundError(
                f"Route-planning mode {mode!r} is unavailable because its checkpoint is missing: {path}"
            )
        if mode == "dqngnn":
            self.route_planning_agent = self.route_planning_agent_dqngnn
        elif mode == "dqn":
            self.route_planning_agent = self.route_planning_agent_dqn
        elif mode == "edqn":
            self.route_planning_agent = self.route_planning_agent_edqn
        else:
            raise ValueError(f"Unknown route planning mode: {mode}")

    def load_route_planning_agents(self, dqngnn_filename = None, dqn_filename = None, edqn_filename = None, *, modes=None):
        self.route_model_paths = {
            "dqngnn": Path(dqngnn_filename) if dqngnn_filename else route_model_path("dqngnn", getattr(self, "environment_name", "env")),
            "dqn": Path(dqn_filename) if dqn_filename else route_model_path("dqn", getattr(self, "environment_name", "env")),
            "edqn": Path(edqn_filename) if edqn_filename else route_model_path("edqn", getattr(self, "environment_name", "env")),
        }
        agents = {
            "dqngnn": self.route_planning_agent_dqngnn,
            "dqn": self.route_planning_agent_dqn,
            "edqn": self.route_planning_agent_edqn,
        }
        self.available_route_modes = {"dijkstra"}
        for mode, path in self.route_model_paths.items():
            if modes is not None and mode not in modes:
                continue
            if path.is_file():
                if hasattr(self, "nid2idx"):
                    validate_route_bundle(path, getattr(self, "environment_name", "env"), self.nid2idx)
                agents[mode].load_full_model(filename=str(path), buffer_name=None)
                self.available_route_modes.add(mode)

    def load_route_planning_agent(self, filename = "./models/using_all/dqngnn_with_EMA.pt"):
        self.route_planning_agent_dqngnn.load_full_model(filename = filename, buffer_name = None)
        self.route_planning_agent = self.route_planning_agent_dqngnn

    def get_min_waiting_time_cs(self, now_time):
        waiting_times = [max(self.eat[cs] - now_time, 0) for cs in self.cs_loc]
        min_idx = int(np.argmin(waiting_times))
        return self.cs_loc[min_idx]

    def update(self, arrived_time, cs, power):
        """
        確定EV抵達後會進到這邊, 根據 [抵達的時間, CS, 抵達時電量] 去推算waiting time跟charging time
        並且要更新充電站狀態
        """
        # print("origin --------------------------")
        # print(self.usage[cs])
        # print(self.eat[cs])
        # print(self.avai_port[cs])

        """
        1. 算waiting time = eat - 抵達時間 
        """
        waiting_time = max(self.eat[cs] - arrived_time, 0)

        """
        2. 算充電時間
        """
        charging_time = self.charge(power)

        """
        3. 要更新usage, eat, avai_port

        3.1 usage
        若抵達時要等待, self.usage += charging_time
        若不用等待, self.usage = 抵達時間 + charging_time

        3.2 eat
        min(usage)

        3.3 avai_port
        argmin(usage)
        """
        ### 3.1 usage
        if waiting_time > 0:
            self.usage[cs][self.avai_port[cs]] += charging_time
        else:
            self.usage[cs][self.avai_port[cs]] = np.round(arrived_time + charging_time, 2)

        ### 3.2 eat
        self.eat[cs] = min(self.usage[cs])

        ### 3.3 avai_port
        self.avai_port[cs] = np.argmin(self.usage[cs])

        # print("updated --------------------------")
        # print(self.usage[cs])
        # print(self.eat[cs])
        # print(self.avai_port[cs])

        # print(f"arrived time {arrived_time}")
        # print(f"waiting time {waiting_time}")
        # print(f"charging_time {charging_time}")

        return waiting_time, charging_time



 
            
    def charge(self, power):
        """
        根據power(現有電量)計算要花多少時間充電
        - 目前設定每個充電站充電速率一樣，都是15min可以充0.7的電量
        """

        # charging_rate = 2 #每15分鐘 0.7SOC
        power_used = 0.9 - power
        time_spend = np.round((power_used/0.7) * 15*60, 2) #轉換成秒

        # time_spend = int(time_spend)

        return time_spend
