# all actors./critics
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPActor(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) actor network for reinforcement learning.
    has two hidden layers with ReLU activations and an output layer with Tanh activation.
    """
    def __init__(self, state_dim, action_dim, action_bound, hidden_sizes=[256, 256]):
        super(MLPActor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], action_dim)
        self.action_bound = action_bound

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.action_bound * torch.tanh(self.l3(a))
        return a

    def save(self):
        # return the state dict of the model
        return self.state_dict()

class MLPCritic(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) critic network for reinforcement learning.
    has two hidden layers with ReLU activations and an output layer.
    """
    def __init__(self, state_dim, action_dim, value_bound, hidden_sizes=[256, 256]):
        super(MLPCritic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], 1)
        self.value_bound = value_bound
        
    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return torch.clamp(q, self.value_bound[0], self.value_bound[1])

    def save(self):
        # return the state dict of the model
        return self.state_dict()
    
class TwinMLPCritic(nn.Module):
    """
    A Twin Multi-Layer Perceptron (MLP) critic network for reinforcement learning.
    used in Double Q-learning to reduce overestimation bias.
    Each network has two hidden layers with ReLU activations and an output layer.
    """
    def __init__(self, state_dim, action_dim, value_bound, hidden_sizes=[256, 256]):
        super(TwinMLPCritic, self).__init__()
        # Critic 1
        self.l1_1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.l2_1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3_1 = nn.Linear(hidden_sizes[1], 1)
        # Critic 2
        self.l1_2 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.l2_2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3_2 = nn.Linear(hidden_sizes[1], 1)
        self.value_bound = value_bound
        
    def forward(self, state, action):
        # Forward pass through Critic 1
        q1 = F.relu(self.l1_1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2_1(q1))
        q1 = self.l3_1(q1)
        # Forward pass through Critic 2
        q2 = F.relu(self.l1_2(torch.cat([state, action], 1)))
        q2 = F.relu(self.l2_2(q2))
        q2 = self.l3_2(q2)
        return torch.clamp(q1, self.value_bound[0], self.value_bound[1]), torch.clamp(q2, self.value_bound[0], self.value_bound[1])

    def save(self):
        # return the state dict of the model
        return self.state_dict()
    
