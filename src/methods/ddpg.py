import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from methods.abstract_methods import RLMethod
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
class MLPActor(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) actor network for reinforcement learning.
    has two hidden layers with ReLU activations and an output layer with Tanh activation.
    """
    def __init__(self, state_dim, action_dim, action_bound, hidden_sizes=[256, 256]):
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
    
class MLPCritic(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) critic network for reinforcement learning.
    has two hidden layers with ReLU activations and an output layer.
    """
    def __init__(self, state_dim, action_dim, value_bound, hidden_sizes=[256, 256]):
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
    
class DDPG():
    """
    the implementation of Deep Deterministic Policy Gradient algorithm.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float, params={}):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.gamma = params.get("gamma", 0.97)
        self.tau = params.get("tau", 0.005)
        self.actor_lr = params.get("actor_lr", 1e-4)
        self.critic_lr = params.get("critic_lr", 1e-3)
        self.batch_size = params.get("batch_size", 128)
        self.max_action = max_action

        # Actor and Critic networks

        self.actor = MLPActor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = MLPActor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = MLPCritic(state_dim, action_dim, value_bound=[-1e10, 1e10]).to(self.device)
        self.critic_target = MLPCritic(state_dim, action_dim, value_bound=[-1e10, 1e10]).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        # Replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)
        
        # copy
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
    
    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]
        
    def update(self):
        if self.replay_buffer.size < self.batch_size:
            return 0, 0

        #    Sample a batch of transitions from the replay buffer
        state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)        
        # Critic update
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = reward + not_done * self.gamma * self.critic_target(next_state, next_action)

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        action = self.actor.forward(state)
        actor_loss = -self.critic(state, action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)        
        return actor_loss.item(), critic_loss.item()
        
    
    
def mainloop(env_name="Pendulum-v1"):
    algo = DDPG(3, 1, 2.0)
    steps = 10000
    cold_start_steps = 1000
    
    env = gym.make(env_name)
    
    # stat
    reward_history = []
    episode_reward = 0
    episode_actorloss = 0
    episode_criticloss = 0
    reward = 0
    state, _ = env.reset()
    for step in range(steps):
        action = algo.select_action(state)
        
        env_action = action + np.random.normal(0, 0.1, size=action.shape)
        env_action = np.clip(env_action, -algo.max_action, algo.max_action)
        if step < cold_start_steps:
            env_action = np.random.uniform(-algo.max_action, algo.max_action, size=action.shape)
        next_state, reward, terminated, truncated, info = env.step(env_action)
        
        algo.replay_buffer.add(state, env_action, next_state, reward/10., terminated)
        state = next_state
        episode_reward += reward    

        if step >= cold_start_steps:
            actor_loss, critic_loss = algo.update()
            episode_actorloss += actor_loss
            episode_criticloss += critic_loss
        if terminated or truncated:
            reward_history.append(episode_reward)
            state, _ = env.reset()
            print(f"Step: {step}/{steps}, Reward: {reward:.2f}, Episode Reward: {episode_reward:.2f}, Actor Loss: {episode_actorloss:.4f}, Critic Loss: {episode_criticloss:.4f}", end="\n")
            episode_reward = 0
            episode_actorloss = 0
            episode_criticloss = 0

    for i in range(len(reward_history)):
        print(f"Episode {i+1}: Reward: {reward_history[i]:.2f}")

if __name__ == "__main__":
    mainloop()