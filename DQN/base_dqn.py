import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import namedtuple, deque
from torch_geometric.nn import GINEConv
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# 創建網路
class Net(nn.Module):
    def __init__(self, s_size, a_size):  
        super(Net, self).__init__()   
        self.s_size = s_size
        self.a_size = a_size
        self.fc1 = nn.Linear(self.s_size, 100)
        self.fc2 = nn.Linear(100, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        actions_value = self.out(x)
        return actions_value
    
class ReplayMemory_numpy(object):
    def __init__(self, capacity, state_shape, action_shape):
        
        self.capacity = capacity
        self.memory = np.zeros((capacity,), dtype=[('state', np.float32, state_shape), 
                                                   ('action', np.int64, action_shape), 
                                                   ('reward', np.float32, 1),
                                                   ('next_state', np.float32, state_shape), 
                                                   ('done', np.float32, 1),
                                                   ])
        self.position = 0
        self.full = False

    def push(self, *args):
        """存儲轉移到記憶中"""
        self.memory[self.position] = Transition(*args)

        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True

    def sample(self, batch_size):
        """從記憶中採樣"""
        max_index = self.capacity if self.full else self.position
        indices = random.sample(range(max_index), batch_size)
        batch = self.memory[indices]

        state_batch = torch.tensor(np.stack(batch['state']))
        action_batch = torch.tensor(np.stack(batch['action']))

        next_state_batch = torch.tensor(np.stack(batch['next_state']))
        reward_batch = torch.tensor(np.stack(batch['reward']))
        done_batch = torch.tensor(np.stack(batch['done']))

        return state_batch, action_batch, next_state_batch, reward_batch, done_batch
    
    def __len__(self):
        return self.capacity if self.full else self.position

    

class base_DQN(object):
    def __init__(self, s_size, a_size, ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval_net, self.target_net = Net(
            s_size, a_size).to(self.device), Net(s_size, a_size).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.a_size = a_size
        self.LR = 3e-4
        self.EPSILON = 0.9
        self.max_epsilon: float = 1.0
        self.min_epsilon = 0.05
        self.target_replace_iter = 100
        self.BATCH_SIZE = 64
        self.min_replay_size = 64
        self.gamma = 0.99
        self.counter = 0
        self.show = True
        self.greedy_cnt = 0
        # self.writer = SummaryWriter(log_dir=f"runs/{writer_name}_exp1")
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.eval_net.parameters(), "lr": self.LR},
            ]
            )
        self.loss_func = nn.MSELoss()

        self.memory = ReplayMemory_numpy(10000, s_size, 1) # state = len(n) + (max_neighbor * 2)
        self.writer = SummaryWriter(log_dir=f"runs/0308_exp2")


    @torch.no_grad()
    def choose_action(self, state, evaluate = False):
        if random.random() > self.EPSILON or evaluate:  # greedy
            input_x = torch.FloatTensor(np.array(state)).to(self.device)
            q = self.eval_net.forward(input_x)
            return int(q.argmax())
        else:  # random
            return random.choice([i for i in range(self.a_size)])
        
    def update_parameters(self, buffer = None):
        self.eval_net.train()

        state_batch, action_batch, next_state_batch, reward_batch, done_batch  = self.memory.sample(self.BATCH_SIZE)

        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device).to(torch.int64)
        next_state_batch = next_state_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        # print("state_batch shape:", state_batch.shape)
        # print("action_batch shape:", action_batch.shape)
        # print("reward_batch shape:", reward_batch.shape)
        # print("next_state_batch shape:", next_state_batch.shape)
        # print("done_batch shape:", done_batch.shape)
        # sys.exit()

        state_action_values = self.eval_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            q_next = self.target_net(next_state_batch)          # (B,27)
            max_q_next = q_next.max(dim=1, keepdim=True)[0]     # (B,1)
            target = reward_batch + (1.0 - done_batch) * self.gamma * max_q_next

        loss = self.loss_func(state_action_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._target_soft_update(tau=0.005)

        return loss.item()

    def _target_soft_update(self, tau=0.005):
        for tp, p in zip(self.target_net.parameters(), self.eval_net.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)


    def save_para(self, filename = "tmp_file.pt"):
        ckpt = {
            "dqn": self.eval_net.state_dict(),
        }
        torch.save(ckpt, filename)
        print("para saved")

    def load_para(self, filename = "tmp_file.pt", strict=True):
        try:
            ckpt = torch.load(f"{filename}", map_location=self.device, weights_only=True)
        except TypeError:  # 舊版 PyTorch 沒這個參數
            ckpt = torch.load(f"{filename}", map_location=self.device)

        self.eval_net.load_state_dict(ckpt.get("dqn", ckpt["dqn"]), strict=strict)
        print("para loaded")