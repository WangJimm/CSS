from __future__ import annotations
import os, sys
from pathlib import Path
from collections import deque
import torch
import math

# os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
# sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
# os.environ["PATH"] = os.environ["PATH"] + ";" + os.path.join(os.environ["SUMO_HOME"], "bin")




# 1) 找到 SUMO 的 tools
if "SUMO_HOME" not in os.environ:
    raise RuntimeError("Please declare environment variable 'SUMO_HOME'")
tools = Path(os.environ["SUMO_HOME"]) / "tools"
sys.path.append(str(tools))
# 2) 保證用「絕對路徑」讀 sumocfg（避免工作目錄不一致）
HERE = Path(__file__).resolve().parents[2]
PARENT_HERE = HERE.parent ## 由於這個資料夾位置不對，把路徑調回去原本的

# 2.1) 插入使用env環境
using_env = "env"
RUN_DIR = HERE / "env"
cfg_path = str(RUN_DIR / "osm.sumocfg")
net_path = str(RUN_DIR / "env_static_tls.net.xml")
general_veh_flows = [1800] # 先用1800試試看，不然太久了
# general_veh_flows = [1500, 1800, 2100]

# Split vehicle-flow settings.
# training_veh_flows controls training episodes.
# validation_veh_flows controls evaluation during training.
# testing_veh_flows controls final evaluation/testing.
training_veh_flows = [1800]
validation_veh_flows = [1800]
testing_veh_flows = [1500,1800,2100]

# Seed ranges are part of the dataset contract. Keep them next to the split
# definitions so scenario generation and loading cannot drift apart.
TRAINING_SEED_START = 1
TRAINING_SEED_END = 20

# # 2.1) 挑選環境，目前是使用small env
# """
# ### testing small env
# """
# using_env = "small_env"
# RUN_DIR = PARENT_HERE / "small_env"
# cfg_path = str(RUN_DIR / "osm.sumocfg")
# net_path = str(RUN_DIR / "small_env.net.xml")

# general_veh_flows = ["free_0900", "norm_1500", "busy_2100", "heavy_2400"]
# training_veh_flows = ["free_0900", "norm_1500"]
# validation_veh_flows = ["busy_2100"]
# testing_veh_flows = ["heavy_2400"]


MAX_DECISIONS_PER_EV = 50
# Route-planning failures are censored at the same 3000-second horizon used by
# the RP episode runner. This keeps an unfinished route from looking cheaper
# than a completed route while allowing charging-agent training to continue.
ROUTE_FAILURE_HORIZON_SEC = 3000.0
# decision_dist_threshold = 100.0
DECISION_DIST_THRESHOLD = 100.0
ROUTE_PLANNING_MODE = ["dqngnn", "dqn", "dijkstra", "edqn"]

class EVState:
    def __init__(self, n_nodes):
        self.done = False
        self.out = False #超出步數 or 超過範圍 
        self.total_cost = 0.0

        self.last_decision_node = None
        self.prev_decision_time = None
        self.prev_edge = None

        self.data_t = None
        self.recent_nodes = deque(maxlen=4)
        self.ever_visit = torch.zeros(n_nodes, 1)
        self.visit_age = torch.zeros(n_nodes, 1)

        self.last_committed_exit_edge = None
        self.current_exit_edge = None
        self.action_mask = None
        
        self.last_action_nid = None
        self.last_state = None
        self.last_est_time = None

        self.neighbors = None
        self.idx = None
        self.neighbor_edges = None

        self.traffic_light_state = None
        self.routes = []
        self.goal = deque()

        self.time_spend = 0

        # ======================= 後面是在cs_navigation新增的
        self.depart_time = None # 紀錄出發時間
        self.travel_time = None # 紀錄從出發到終點所花的時間
        self.goal_cs = None
        self.decision = 0
        self.power = None
        self.waiting_time = None
        self.charging_time = None
        self.sacd_state = None
        self.action = None
        self.feasible_cs_actions = None
        self.planned_sumo_route = None
        self.failure_reason = None

class EMA_feature:
    def __init__(self, half_life_sec: float | None = None, tau_sec: float | None = None):
        assert (half_life_sec is None) ^ (tau_sec is None), "給 half_life 或 tau 其一"
        self.half_life = half_life_sec
        self.tau = tau_sec
        self.val = {}       # edge_id -> ema value
        self.last_t = {}    # edge_id -> last update time

    def _alpha(self, dt: float) -> float:
        if self.half_life is not None:
            return 1 - math.pow(0.5, dt / self.half_life)
        else:
            return 1 - math.exp(-dt / self.tau)
        
    def update(self, key, x: float, now_t: float) -> float:
        if key not in self.val:
            self.val[key] = x
            self.last_t[key] = now_t
            return self.val[key]
        dt = max(0.0, now_t - self.last_t[key])
        a = self._alpha(dt)
        # a = 1           ## test for no EMA
        self.val[key] = a * x + (1 - a) * self.val[key]
        self.last_t[key] = now_t
        return self.val[key]
