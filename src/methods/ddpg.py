import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from methods.abstract_methods import RLMethod
import torch.nn.functional as F


class QValueNetWork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[256, 256]):
        super(QValueNetWork, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        q_value = self.l3(x)
        return q_value
from utils.ReplayBuffer import ReplayBuffer

class DDPG(RLMethod):
    """
    the implementation of Deep Deterministic Policy Gradient algorithm.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float, params=None):
        super(DDPG, self).__init__(state_dim, action_dim, max_action, params)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.gamma = params.get("gamma", 0.99)
        self.tau = params.get("tau", 0.005)
        self.actor_lr = params.get("actor_lr", 1e-4)
        self.critic_lr = params.get("critic_lr", 1e-3)
        self.batch_size = params.get("batch_size", 64)
        self.max_action = max_action

        # Actor and Critic networks
        from dnpart.actor import MLPActor, MLPCritic

        self.actor = MLPActor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = MLPActor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = MLPCritic(state_dim, action_dim, value_bound=[-1e10, 1e10]).to(self.device)
        self.critic_target = MLPCritic(state_dim, action_dim, value_bound=[-1e10, 1e10]).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # Replay buffer

        self.replay_buffer = ReplayBuffer(params.get("buffer_size", int(1e6)), state_dim, action_dim, self.device)