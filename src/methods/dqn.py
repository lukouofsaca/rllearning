# an dqn file for RL
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm
import os
from methods.registry import register_method
from methods.abstract_methods import RLMethod


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.l3(x)

@register_method("ppo")
class DQN(RLMethod):
    def __init__(self, state_dim, action_dim, max_action, 
                 actor=None,critic=None,
                     params=None,):
        super().__init__(state_dim, action_dim, max_action, actor, critic, params)
        if actor is None:
            self.actor = Actor(state_dim, action_dim, max_action)
        if critic is None:
            self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params['critic_lr'])
        self.it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            dist = self.actor.get_dist(state)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(state)
        
        # 裁剪动作到环境允许范围
        action = torch.clamp(action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten(), action_logprob.cpu().data.numpy().flatten(), value.cpu().data.numpy().flatten()
    
    def update(self):
        action, value = self.select_action(state)
        # q-learning update logic here
        
        # update actor and critic networks
        # ...

    def save(self, path):
        pass

    def load(self, path):
        pass