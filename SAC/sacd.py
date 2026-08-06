"""
專門處理充電站選擇
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from .utils import MinSegmentTree, SumSegmentTree
from typing import Dict, List, Tuple
import sys
from torch.utils.tensorboard import SummaryWriter


# ---------------------------
# 離散 Actor：輸出各離散動作的 logits，並以 softmax 得到概率
# ---------------------------
class DiscreteActor(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=256):
        super(DiscreteActor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, n_actions)
    
    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        logits = self.logits(x)
        return logits
    def get_policy(self, state):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs
    
    def sample(self, state):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, probs
    


# ---------------------------
# 離散 Critic：Twin Q-Network，針對每個 state 輸出所有動作的 Q 值
# ---------------------------
class DiscreteCritic(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=256):
        super(DiscreteCritic, self).__init__()
        # 第一個 Q 網絡
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, n_actions)
        # 第二個 Q 網絡
        self.l3 = nn.Linear(state_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, n_actions)
    
    def forward(self, state):
        x1 = F.relu(self.l1(state))
        x1 = F.relu(self.l2(x1))
        q1 = self.q1(x1)
        
        x2 = F.relu(self.l3(state))
        x2 = F.relu(self.l4(x2))
        q2 = self.q2(x2)
        return q1, q2
    
# --- Prioritized Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        batch = list(zip(*samples))
        states = torch.tensor(np.array(batch[0]), dtype=torch.float32)
        actions = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32)
        dones = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

# Replay Buffer 定義
class ReplayBufferGPT:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # 可以在這裡先做一點資料格式整理，比如 squeeze 多餘的維度
        state = np.array(state).squeeze()
        next_state = np.array(next_state).squeeze()
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    
    def sample(self, batch_size):
        indices = random.sample(range(len(self.states)), batch_size)

        batch_states = np.array([self.states[i] for i in indices])
        batch_actions = np.array([self.actions[i] for i in indices])
        batch_rewards = np.array([self.rewards[i] for i in indices])
        batch_next_states = np.array([self.next_states[i] for i in indices])
        batch_dones = np.array([self.dones[i] for i in indices])
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
    
    def __len__(self):
        return len(self.states)
class CompactReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, env_state, action, reward, next_env_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (env_state, action, reward, next_env_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return list(zip(*batch))  # 回傳為多個 list（依序為每個欄位）

    def __len__(self):
        return len(self.buffer)
    

class DiscreteSACAgent:
    def __init__(self, state_dim, n_actions, using_graph = False, edge_index = None, hidden_dim=256, discount=0.99, tau=0.005,
                 lr=3e-4, batch_size = 32, min_replay_size = 1000, 
                 PER = False, per_alpha = 0.2, per_beta = 0.6, prior_eps = 1e-6, 
                 REPLAY_BUFFER_SIZE = 100000):
        
        self.min_replay_size = min_replay_size
        self.batch_size = batch_size
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.discount = discount
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = DiscreteActor(state_dim, n_actions, hidden_dim).to(self.device)
        self.critic = DiscreteCritic(state_dim, n_actions, hidden_dim).to(self.device)
        self.critic_target = DiscreteCritic(state_dim, n_actions, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.target_entropy = 0.98 * np.log(n_actions)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        self.writer = SummaryWriter(log_dir=f"runs/0308_exp2")
        self.learn_step_counter = 0



    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def choose_action(self, sacd_state, eval = False):
        state = torch.FloatTensor(sacd_state).to(self.device).unsqueeze(0)
        if eval:
            with torch.no_grad():
                logits = self.actor(state)
                probs = F.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1)
            return action.item()
        else:
            with torch.no_grad():
                action, _, _ = self.actor.sample(state)

        return action.item()
    
    def update_parameters(self, replay_buffer):
        """
        更新 Actor、Critic 與 熱度參數。
        batch_state: (B, state_dim) FloatTensor
        batch_action: (B, 1) LongTensor
        batch_reward: (B, 1) FloatTensor
        batch_next_state: (B, state_dim) FloatTensor
        batch_done: (B, 1) FloatTensor (0 或 1)
        """
        self.learn_step_counter += 1
        state, action, reward, next_state, done = replay_buffer.sample(self.batch_size)
        # state = torch.FloatTensor(state).to(self.device)
        # next_state = torch.FloatTensor(next_state).to(self.device)
        # action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        # reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        # done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        state = torch.as_tensor(np.array(state), dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(np.array(action), dtype=torch.int64, device=self.device).view(-1, 1)
        reward = torch.as_tensor(np.array(reward), dtype=torch.float32, device=self.device).view(-1, 1)
        done = torch.as_tensor(np.array(done), dtype=torch.float32, device=self.device).view(-1, 1)


        alpha = self.log_alpha.exp()

        # print(f"state {state.shape}")
        # print(f"next_state {next_state.shape}")
        # print(f"action {action.shape}")
        # print(f"reward {reward.shape}")
        # print(f"done {done.shape}")

        # =========================================================
        # 1) Critic update
        # =========================================================
        with torch.no_grad():
            next_probs, next_log_probs = self.actor.get_policy(next_state)           # (B, A), (B, A)
            next_q1_target, next_q2_target = self.critic_target(next_state)         # (B, A), (B, A)
            next_min_q_target = torch.min(next_q1_target, next_q2_target)           # (B, A)

            next_v = (next_probs * (next_min_q_target - alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = reward + (1.0 - done) * self.discount * next_v

        current_q1, current_q2 = self.critic(state)                                 # (B, A), (B, A)
        current_q1 = current_q1.gather(1, action)                                   # (B, 1)
        current_q2 = current_q2.gather(1, action)  

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # =========================================================
        # 2) Actor update
        # =========================================================
        probs, log_probs = self.actor.get_policy(state)                             # (B, A), (B, A)
        q1, q2 = self.critic(state)                                                 # (B, A), (B, A)
        min_q = torch.min(q1, q2)                                                   # (B, A)

        actor_loss = (probs * (alpha.detach() * log_probs - min_q)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # =========================================================
        # 3) Alpha update
        # =========================================================
        entropy = -(probs * log_probs).sum(dim=1).mean()

        alpha_loss = -(self.log_alpha * (self.target_entropy - entropy.detach())).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # =========================================================
        # 4) Soft update target critic
        # =========================================================
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        self.writer.add_scalar("loss/critic_loss", critic_loss.item(), self.learn_step_counter)
        self.writer.add_scalar("loss/actor_loss", actor_loss.item(), self.learn_step_counter)
        self.writer.add_scalar("loss/alpha_loss", alpha_loss.item(), self.learn_step_counter)
        self.writer.add_scalar("stats/alpha", alpha.item(), self.learn_step_counter)


        return critic_loss.item(), actor_loss.item(), alpha_loss.item(), alpha.item()

    def save_para(self, filename = "tmp_file.pt"):
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }
        torch.save(ckpt, filename)
        print("para saved")

    def load_para(self, filename = "tmp_file.pt", strict = True):
        try:
            ckpt = torch.load(filename, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(filename, map_location=self.device)

        self.actor.load_state_dict(ckpt["actor"], strict=strict)
        self.critic.load_state_dict(ckpt["critic"], strict=strict)
        self.critic_target.load_state_dict(ckpt["critic_target"], strict=strict)

        if "actor_optimizer" in ckpt:
            self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        if "critic_optimizer" in ckpt:
            self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        if "alpha_optimizer" in ckpt:
            self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        if "log_alpha" in ckpt:
            self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))

        print("para loaded")



def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
