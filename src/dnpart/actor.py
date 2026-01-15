# all actors./critics
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPActor(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) actor network for reinforcement learning.
    has two hidden layers with ReLU activations and an output layer with Tanh activation.
    """
    def __init__(self, state_dim, action_dim, action_bound, hidden_sizes=[64, 64]):
        super(MLPActor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.lm = nn.LayerNorm(hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], action_dim)
        self.action_bound = action_bound

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = self.lm(a)
        a = F.relu(self.l2(a))
        a = self.action_bound * torch.tanh(self.l3(a))
        return a

    def save(self):
        # return the state dict of the model
        return self.state_dict()
    
    

class MuThiActor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, hidden_sizes=[64, 64]):
        super(MuThiActor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.mu = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        
        self.action_bound = action_bound
    
    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mu = self.mu(x)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std
    
    def sample(self, state):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        action = self.action_bound * torch.tanh(action)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob
            
    def calculate_log_prob(self, state, action):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return log_prob
    
    

class MLPCritic(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) critic network for reinforcement learning.
    has two hidden layers with ReLU activations and an output layer.
    """
    def __init__(self, state_dim, action_dim, value_bound, hidden_sizes=[64, 64]):
        super(MLPCritic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.lm = nn.LayerNorm(hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], 1)
        self.value_bound = value_bound
        
    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = self.lm(q)
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

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
    
