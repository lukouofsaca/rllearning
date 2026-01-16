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
        """
        max_action is useless now, just for compatibility.
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if params is None:
            params = {}
        self.params = params
        self.action_dim = action_dim
        self.max_action = max_action
        # DQN Hyperparameters
        self.gamma = params.get("gamma", 0.99)
        self.tau = params.get("tau", 0.005)
        self.lr = params.get("lr", 3e-4)
        self.batch_size = params.get("batch_size", 2)
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


        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
            action = q_values.argmax(dim=1).cpu().numpy()        
                
        return action, None, None

    def store_transition(self, transition):
        state, next_state, action, reward, done = transition
        

        self.replay_buffer.add(state, action, next_state, reward, done)

    def update(self):
        if self.replay_buffer.size < self.batch_size:
            return
        state, action, next_state, reward, done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            target_Q = self.target_q_net(next_state).max(1, keepdim=True)[0]
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q = self.q_net(state).gather(1, action.long())
        
        # print(current_Q, target_Q)
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
        


if __name__ == "__main__":
    import gymnasium as gym
    from utils.ActionWrapper import EpsilonGreedyActionWrapper

    env_name = "CartPole-v1"
    steps = 100000
    minibatch_size = 64
    cold_start_steps = 1000
    epsilon = 0.1
    
    
    env = gym.make(env_name)
    
    ## 假设单维离散动作空间
    action_space = env.action_space
    action_space_dim = action_space.n    
    action_dim = 1
    
    state_dim = env.observation_space.shape[0]
    
    network = DQN(state_dim, action_space_dim, None)
    
    state , _ = env.reset()
    
    action_wrapper = EpsilonGreedyActionWrapper(epsilon, action_dim, action_space)
    
    def select_action(network, state):
        action, _, _ = network.select_action(state)
        # to action space
        action = action_wrapper(action)
        if isinstance(action, np.ndarray):
            action = action.item()
        return action
    
    
    reward_history = []
    episodereward = 0
    episode = 0
    for i in range(steps):
        # cold start
        
        if i < cold_start_steps:
            action = env.action_space.sample()
        else:
            action = select_action(network, state)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated
        network.store_transition((state, next_state, action, reward, done))
        state = next_state
        # reset network
        episodereward += reward
        if terminated or truncated:
            state, _ = env.reset()
            episode += 1
            reward_history.append(episodereward)
            episodereward = 0
            # print(f"Step: {i}, Episode Reward: {reward_history[-1]:.2f}, Epsilon: {network.epsilon:.3f}")
    
        # update network
        if i >= cold_start_steps:
            for j in range(1):  
                network.update()
        if (i + 1) % 1000 == 0:
            avg_reward = np.mean(reward_history[-10:]) if len(reward_history) >=10 else np.mean(reward_history)
            print(f"Step: {i+1}, Episode: {episode}, Avg Reward(10): {avg_reward:.2f}, Epsilon: {network.epsilon:.3f}")
                
    
            