import copy
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.tensorboard import SummaryWriter

from css.graph import build_data_runtime, freeze_pyg_data


DEFAULT_PAPER_BASELINE_CHECKPOINT = Path(__file__).resolve().parents[3] / "parameters" / "env_paper_baseline.pt"


def _to_scalar(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x).reshape(-1)
        return arr[0].item()
    return x


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        transition = (
            copy.deepcopy(state),
            int(_to_scalar(action)),
            float(_to_scalar(reward)),
            copy.deepcopy(next_state),
            float(_to_scalar(done)),
        )
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        valid_len = len(self.buffer)
        prios = self.priorities[:valid_len]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(valid_len, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        weights = (valid_len * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(prio)

    def __len__(self):
        return len(self.buffer)


class GCNEncoder(nn.Module):
    def __init__(self, node_dim, hidden_dim, dropout_p=0.1):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout_p = dropout_p

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x


class ModifiedRainbowNetwork(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_nodes, action_dim, dropout_p=0.1):
        super().__init__()
        self.encoder = GCNEncoder(node_dim, hidden_dim, dropout_p=dropout_p)
        flattened_dim = hidden_dim * num_nodes
        self.flatten_head = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, action_dim),
        )
        self.action_dim = action_dim

    def forward_state(self, state):
        data = state["graph"].to(next(self.parameters()).device)
        node_embeddings = self.encoder(data.x, data.edge_index)
        flat = node_embeddings.reshape(1, -1)
        latent = self.flatten_head(flat)
        value = self.value_stream(latent)
        advantage = self.advantage_stream(latent)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values.squeeze(0)


@dataclass
class PaperStateConfig:
    time_scale: float = 3600.0
    power_scale: float = 1.0
    ev_count_scale: float = 50.0
    waiting_scale: float = 3600.0
    length_scale: float = 1000.0
    resistance_scale: float = 300.0


class PaperBaselineAgent:
    uses_graph_state = True

    def __init__(
        self,
        env,
        hidden_dim=128,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=32,
        min_replay_size=128,
        replay_capacity=20000,
        per_alpha=0.6,
        per_beta=0.4,
        epsilon=0.9,
        min_epsilon=0.05,
        epsilon_decay=0.995,
        dropout_p=0.1,
        log_dir=None,
    ):
        self.env = env
        self.action_dim = env.action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.min_replay_size = min_replay_size
        self.PER_BETA = per_beta
        self.prior_eps = 1e-6
        self.EPSILON = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_cfg = PaperStateConfig()

        node_dim = 10
        num_nodes = len(env.nid2idx)
        self.eval_net = ModifiedRainbowNetwork(node_dim, hidden_dim, num_nodes, self.action_dim, dropout_p=dropout_p).to(self.device)
        self.target_net = ModifiedRainbowNetwork(node_dim, hidden_dim, num_nodes, self.action_dim, dropout_p=dropout_p).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(replay_capacity, alpha=per_alpha)
        self.writer = SummaryWriter(log_dir=log_dir or "runs/paper_baseline_rainbow")
        self.learn_step_counter = 0

    def _build_node_feature_matrix(self, runtime_graph, depart_time, depart_node, depart_power):
        node_count = runtime_graph.x.size(0)
        features = torch.zeros((node_count, 10), dtype=torch.float32)

        # Paper-style expanded node feature:
        # X^{t,i} = [E^{t,i}, C^{t,i}, T^{t,i}]
        # EV nodes use [t_o, p_o, e, N_ev], CS nodes use [C_k, P_k, omega_k, N_w],
        # and TN nodes use [l_mn, wt_mn], while the remaining blocks are zero-padded.

        # EV block: [departure_time, current_position_flag, battery, active_ev_count]
        current_idx = self.env.nid2idx[depart_node]
        features[current_idx, 0] = depart_time / self.feature_cfg.time_scale
        features[current_idx, 1] = 1.0
        features[current_idx, 2] = float(depart_power) / self.feature_cfg.power_scale
        features[current_idx, 3] = len(getattr(self.env, "evs", [])) / self.feature_cfg.ev_count_scale

        # CS block: [num_piles, charging_power, waiting_time, waiting_ev_count]
        for cs in self.env.cs_loc:
            cs_idx = self.env.nid2idx[cs]
            features[cs_idx, 4] = float(self.env.cs_pile)
            features[cs_idx, 5] = float(self.env.max_power) / 100.0
            features[cs_idx, 6] = max(self.env.eat[cs] - depart_time, 0.0) / self.feature_cfg.waiting_scale
            waiting_ev_count = sum(1 for end_t in self.env.usage[cs] if end_t > depart_time)
            features[cs_idx, 7] = waiting_ev_count / self.feature_cfg.ev_count_scale

        # TN block: [segment length proxy, road resistance proxy]
        edge_index = runtime_graph.edge_index
        edge_attr = runtime_graph.edge_attr
        for idx in range(node_count):
            mask = edge_index[1] == idx
            if mask.any():
                incident_edge_attr = edge_attr[mask]
                features[idx, 8] = incident_edge_attr[:, 3].mean()
                features[idx, 9] = incident_edge_attr[:, 0].mean()

        return features

    def build_state(self, depart_time, depart_node, depart_power, travel_time, mean_speed, occupancy):
        runtime_graph = build_data_runtime(self.env.sumo_static_pack, travel_time, mean_speed, occupancy)
        runtime_graph = freeze_pyg_data(runtime_graph)
        runtime_graph.x = self._build_node_feature_matrix(runtime_graph, depart_time, depart_node, depart_power)
        return {
            "graph": runtime_graph,
        }

    @torch.no_grad()
    def choose_action(self, state, eval=False):
        if (not eval) and random.random() < self.EPSILON:
            return random.randrange(self.action_dim)
        q_values = self.eval_net.forward_state(state)
        return int(torch.argmax(q_values).item())

    def update_parameters(self, replay_buffer):
        self.learn_step_counter += 1
        samples, indices, weights = replay_buffer.sample(self.batch_size, beta=self.PER_BETA)
        weights = weights.to(self.device).view(-1, 1)

        chosen_q_values = []
        target_q_values = []

        for state, action, reward, next_state, done in samples:
            q = self.eval_net.forward_state(state)
            chosen_q_values.append(q[action].view(1, 1))
            reward_t = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
            done_t = torch.tensor([[done]], dtype=torch.float32, device=self.device)
            if next_state is None or done >= 1.0:
                target_q_values.append(reward_t)
            else:
                with torch.no_grad():
                    next_q_eval = self.eval_net.forward_state(next_state)
                    next_action = int(torch.argmax(next_q_eval).item())
                    next_q_target = self.target_net.forward_state(next_state)
                    target_v = next_q_target[next_action].view(1, 1)
                    target_q_values.append(reward_t + (1.0 - done_t) * self.gamma * target_v)

        chosen_q = torch.cat(chosen_q_values, dim=0)
        target_q = torch.cat(target_q_values, dim=0)

        loss_each = F.smooth_l1_loss(chosen_q, target_q, reduction="none")
        critic_loss = (loss_each * weights).mean()

        self.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        td_errors = (chosen_q.detach() - target_q).abs().view(-1).cpu().numpy() + self.prior_eps
        replay_buffer.update_priorities(indices, td_errors)
        self._soft_update(self.target_net, self.eval_net, self.tau)

        self.writer.add_scalar("loss/q_loss", critic_loss.item(), self.learn_step_counter)
        self.writer.add_scalar("stats/epsilon", self.EPSILON, self.learn_step_counter)
        return critic_loss.item()

    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save_para(self, filename="paper_baseline_tmp.pt"):
        ckpt = {
            "eval_net": self.eval_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.EPSILON,
            "learn_step_counter": self.learn_step_counter,
        }
        torch.save(ckpt, filename)
        print("para saved")

    def load_para(self, filename=DEFAULT_PAPER_BASELINE_CHECKPOINT, strict=True):
        try:
            ckpt = torch.load(filename, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(filename, map_location=self.device)
        self.eval_net.load_state_dict(ckpt["eval_net"], strict=strict)
        self.target_net.load_state_dict(ckpt["target_net"], strict=strict)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.EPSILON = ckpt.get("epsilon", self.EPSILON)
        self.learn_step_counter = ckpt.get("learn_step_counter", 0)
        print("para loaded")
