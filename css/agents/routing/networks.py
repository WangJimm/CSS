from __future__ import annotations

from dataclasses import dataclass
import random
from collections import namedtuple
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv


Transition = namedtuple(
    "Transition",
    (
        "state",
        "action",
        "next_state",
        "reward",
        "done",
        "next_action_mask",
        "data_t",
        "next_data_t",
        "bag",
        "next_bag",
    ),
)


@dataclass(frozen=True)
class DQNGNNShapeConfig:
    num_nodes: int
    max_neighbors: int
    node_in: int
    edge_in: int
    gnn_hidden: int = 128

    @property
    def tls_state_dim(self) -> int:
        return self.max_neighbors * 2

    @property
    def graph_embedding_dim(self) -> int:
        return self.gnn_hidden * (self.max_neighbors + 2)

    @property
    def dqn_input_dim(self) -> int:
        return self.graph_embedding_dim + self.tls_state_dim


class GraphEncoderGINE(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int = 128, num_layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_channels = node_in if layer_idx == 0 else hidden
            mlp = nn.Sequential(
                nn.Linear(in_channels, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.layers.append(GINEConv(mlp, train_eps=True, edge_dim=edge_in))
            self.norms.append(nn.LayerNorm(hidden))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, norm in zip(self.layers, self.norms):
            residual = h
            h = F.relu(conv(h, edge_index, edge_attr))
            if residual.shape == h.shape:
                h = h + residual
            h = norm(h)
        return h


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class PrioritizedReplayMemoryNumpy:
    def __init__(
        self,
        capacity: int,
        state_shape: int | tuple[int, ...],
        action_shape: int | tuple[int, ...],
        action_mask_shape: int,
        *,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 1e-5,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.sample_count = 0
        self.memory = np.zeros(
            (capacity,),
            dtype=[
                ("state", np.float32, state_shape),
                ("action", np.int64, action_shape),
                ("next_state", np.float32, state_shape),
                ("reward", np.float32, 1),
                ("done", np.float32, 1),
                ("next_action_mask", np.bool_, (action_mask_shape,)),
                ("data_t", object),
                ("next_data_t", object),
                ("bag", object),
                ("next_bag", object),
            ],
        )
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.full = False

    def push(self, *args: Any) -> None:
        self.memory[self.position] = Transition(*args)
        max_priority = self.priorities.max() if len(self) else 1.0
        self.priorities[self.position] = max(max_priority, 1.0)
        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True

    def sample(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[Any], list[Any], list[Any], list[Any], np.ndarray, torch.Tensor]:
        max_index = self.capacity if self.full else self.position
        priorities = self.priorities[:max_index]
        if priorities.sum() <= 0:
            probabilities = np.full(max_index, 1.0 / max_index, dtype=np.float32)
        else:
            scaled = priorities ** self.alpha
            probabilities = scaled / scaled.sum()
        indices = np.random.choice(max_index, batch_size, replace=False, p=probabilities)
        batch = self.memory[indices]
        self.sample_count += 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.sample_count / self.beta_frames)
        weights = (max_index * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        return (
            torch.tensor(np.stack(batch["state"])),
            torch.tensor(np.stack(batch["action"])),
            torch.tensor(np.stack(batch["next_state"])),
            torch.tensor(np.stack(batch["reward"])),
            torch.tensor(np.stack(batch["done"])),
            torch.tensor(np.stack(batch["next_action_mask"])),
            [item for item in batch["data_t"]],
            [item for item in batch["next_data_t"]],
            [item for item in batch["bag"]],
            [item for item in batch["next_bag"]],
            indices,
            torch.tensor(weights, dtype=torch.float32).unsqueeze(1),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: torch.Tensor) -> None:
        errors = td_errors.detach().abs().flatten().cpu().numpy()
        for idx, error in zip(indices, errors):
            self.priorities[idx] = float(error + self.epsilon)

    def __len__(self) -> int:
        return self.capacity if self.full else self.position

