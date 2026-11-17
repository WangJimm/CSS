from __future__ import annotations

import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


class CompetitiveNet(nn.Module):
    def __init__(self, s_size: int, a_size: int, hidden_dims=(256, 128, 64, 16)):
        super().__init__()
        layers = []
        prev = s_size
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        self.feature = nn.Sequential(*layers)
        self.value = nn.Linear(prev, 1)
        self.advantage = nn.Linear(prev, a_size)

    def forward(self, x):
        h = self.feature(x)
        value = self.value(h)
        advantage = self.advantage(h)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class ReplayMemory_numpy(object):
    def __init__(self, capacity, state_shape, action_shape):
        self.capacity = capacity
        state_shape = (state_shape,) if isinstance(state_shape, int) else tuple(state_shape)
        action_shape = (action_shape,) if isinstance(action_shape, int) else tuple(action_shape)
        self.memory = np.zeros((capacity,), dtype=[('state', np.float32, state_shape),
                                                   ('action', np.int64, action_shape),
                                                   ('reward', np.float32, (1,)),
                                                   ('next_state', np.float32, state_shape),
                                                   ('done', np.float32, (1,)),
                                                   ])
        self.position = 0
        self.full = False

    def push(self, *args):
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True

    def sample(self, batch_size):
        max_index = self.capacity if self.full else self.position
        indices = random.sample(range(max_index), batch_size)
        batch = self.memory[indices]
        return (
            torch.tensor(np.stack(batch['state'])),
            torch.tensor(np.stack(batch['action'])),
            torch.tensor(np.stack(batch['next_state'])),
            torch.tensor(np.stack(batch['reward'])),
            torch.tensor(np.stack(batch['done'])),
        )

    def __len__(self):
        return self.capacity if self.full else self.position


class base_eDQN(object):
    """CSS eDQN agent: Double target, competitive Q network, and Huber loss."""

    def __init__(self, s_size, a_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_net = CompetitiveNet(s_size, a_size).to(self.device)
        self.target_net = CompetitiveNet(s_size, a_size).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.a_size = a_size
        self.LR = 1e-5
        self.EPSILON = 0.9
        self.max_epsilon = 1.0
        self.min_epsilon = 0.05
        self.target_replace_iter = 100
        self.BATCH_SIZE = 128
        self.batch_size = self.BATCH_SIZE
        self.min_replay_size = self.BATCH_SIZE
        self.gamma = 0.95
        self.counter = 0
        self.show = True
        self.greedy_cnt = 0
        self.huber_delta = 1.0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.LR)
        self.memory = ReplayMemory_numpy(10000, s_size, 1)
        self.writer = _NullWriter()

    @torch.no_grad()
    def choose_action(self, state, evaluate=False):
        if random.random() > self.EPSILON or evaluate:
            input_x = torch.FloatTensor(np.array(state)).to(self.device)
            q = self.eval_net.forward(input_x)
            return int(q.argmax())
        return random.choice([i for i in range(self.a_size)])

    def _huber_loss(self, pred, target):
        error = pred - target
        abs_error = error.abs()
        quadratic = torch.minimum(abs_error, torch.tensor(self.huber_delta, device=self.device))
        linear = abs_error - quadratic
        return 0.5 * quadratic.pow(2) + self.huber_delta * linear

    def update_parameters(self, buffer=None):
        self.eval_net.train()
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = self.memory.sample(self.BATCH_SIZE)

        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device).to(torch.int64)
        next_state_batch = next_state_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        state_action_values = self.eval_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_eval_q = self.eval_net(next_state_batch)
            next_action = next_eval_q.argmax(dim=1, keepdim=True)
            next_target_q = self.target_net(next_state_batch)
            selected_next_q = next_target_q.gather(1, next_action)
            target = reward_batch + (1.0 - done_batch) * self.gamma * selected_next_q

        loss = self._huber_loss(state_action_values, target).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.counter += 1
        if self.counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        return loss.item()

    def save_para(self, filename="tmp_file.pt"):
        ckpt = {
            "edqn": self.eval_net.state_dict(),
            "edqn_target": self.target_net.state_dict(),
            "agent_type": "css_edqn",
        }
        torch.save(ckpt, filename)
        print("para saved")

    def load_para(self, filename="tmp_file.pt", strict=True):
        try:
            ckpt = torch.load(filename, map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(filename, map_location=self.device)
        state = ckpt.get("edqn", ckpt.get("dqn"))
        if state is None:
            raise KeyError("checkpoint does not contain 'edqn' or 'dqn'")
        self.eval_net.load_state_dict(state, strict=strict)
        target_state = ckpt.get("edqn_target")
        if target_state is not None:
            self.target_net.load_state_dict(target_state, strict=strict)
        else:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        print("para loaded")

