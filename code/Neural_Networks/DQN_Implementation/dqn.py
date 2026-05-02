"""Reusable DQN model components for traffic control experiments."""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return tuple(zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


def select_epsilon_greedy_action(model: DQN, state, action_dim: int, epsilon: float) -> int:
    """Choose a random action during exploration, otherwise the highest-Q action."""

    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)

    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def optimize_dqn(
    model: DQN,
    target_model: DQN,
    optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
):
    """Run one DQN update and return the scalar loss, or None before warmup."""

    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.long)
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32)
    dones = torch.as_tensor(dones, dtype=torch.float32)

    current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_actions = model(next_states).argmax(dim=1)
        next_q = target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards + gamma * next_q * (1.0 - dones)

    loss = F.smooth_l1_loss(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())
