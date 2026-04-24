"""Reusable DQN model components for traffic control experiments."""

from __future__ import annotations

import random
from collections import deque

import torch.nn as nn


class DQN(nn.Module):
    """Small fully connected Q-network for the traffic environment."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity: int = 10000) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return states, actions, rewards, next_states

    def __len__(self) -> int:
        return len(self.buffer)
