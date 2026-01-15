# an dqn file for RL
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from methods.registry import register_method
from methods.abstract_methods import RLMethod
from utils.ReplayBuffer import ReplayBuffer

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.l3(x)

@register_method("dqn")
class DQN(RLMethod):
    def __init__(self, state_dim, action_dim, max_action, params=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if params is None:
            params = {}
        self.params = params
        self.action_dim = action_dim
        
        # DQN Hyperparameters
        self.gamma = params.get("gamma", 0.99)
        self.tau = params.get("tau", 0.005)
        self.lr = params.get("lr", 3e-4)
        self.batch_size = params.get("batch_size", 256)
        self.epsilon = params.get("epsilon", 1.0)
        self.epsilon_decay = params.get("epsilon_decay", 0.995)
        self.epsilon_min = params.get("epsilon_min", 0.01)

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_net = copy.deepcopy(self.q_net)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        # For DQN, we store action indices, so action dimension in buffer is 1
        self.replay_buffer = ReplayBuffer(state_dim, 1, max_size=int(1e6))
        self.it = 0

    def select_action(self, state):
        # state: (num_envs, state_dim) or (state_dim,)
        if isinstance(state, tuple):
            state = state[0]
        
        # Handle batch input
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
            batch_size = 1
        else:
            batch_size = state.shape[0]

        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_dim, size=batch_size)
        else:
            state = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state)
                action = q_values.argmax(dim=1).cpu().numpy()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return action, None, None

    def store_transition(self, transition):
        state, next_state, action, action_logprob, reward, done, value = transition
        
        # Check if inputs are batches
        if len(state.shape) > 1 and state.shape[0] > 1:
            # Iterate over batch
            for i in range(state.shape[0]):
                self.replay_buffer.add(state[i], action[i], next_state[i], reward[i], done[i])
        else:
            # Single transition
            # Ensure scalar/1D array consistency
            s = state if len(state.shape) == 1 else state[0]
            ns = next_state if len(next_state.shape) == 1 else next_state[0]
            a = action if np.isscalar(action) or action.ndim==0 else action[0]
            r = reward if np.isscalar(reward) or reward.ndim==0 else reward[0]
            d = done if np.isscalar(done) or done.ndim==0 else done[0]
            self.replay_buffer.add(s, a, ns, r, d)

    def update(self):
        if self.replay_buffer.size < self.batch_size:
            return

        state, action, next_state, reward, done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            target_Q = self.target_q_net(next_state).max(1, keepdim=True)[0]
            target_Q = reward + (1-done) * self.gamma * target_Q

        current_Q = self.q_net(state).gather(1, action.long())

        loss = F.mse_loss(current_Q, target_Q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        for param, target_param in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        self.it += 1

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
