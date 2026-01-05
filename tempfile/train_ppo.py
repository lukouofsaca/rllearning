import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import numpy as np

# --- 1. 定义网络结构 ---
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.mu_head = nn.Linear(64, action_dim)
        self.sigma_head = nn.Linear(64, action_dim)

    def forward(self, x):
        x = self.fc(x)
        mu = 2.0 * torch.tanh(self.mu_head(x)) # Pendulum 动作范围是 [-2, 2]
        sigma = F.softplus(self.sigma_head(x)) + 1e-3 # 确保标准差为正
        return mu, sigma

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x):
        return self.fc(x)

# --- 2. PPO 算法主体 ---
class PPO:
    def __init__(self, state_dim, action_dim):
        self.actor = PolicyNet(state_dim, action_dim)
        self.critic = ValueNet(state_dim)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=2e-4)
        self.clip_param = 0.2
        self.gamma = 0.9
        self.mu = 0
        self.std = 0.1
        
    def select_action(self, state):
        state = torch.FloatTensor(state)
        mu, sigma = self.actor(state)
        self.mu = mu
        self.std = sigma
        dist = Normal(mu, sigma)
        action = dist.sample()
        # 限制动作范围并记录 log_prob
        action_log_prob = dist.log_prob(action)
        return action.detach().numpy(), action_log_prob.detach()

    def update(self, memory):
        # 转换 memory 中的列表为 Tensor
        states = torch.FloatTensor(np.array(memory['states']))
        actions = torch.FloatTensor(np.array(memory['actions']))
        old_log_probs = torch.stack(memory['log_probs'])
        rewards = memory['rewards']
        
        # 计算折扣奖励 (Returns)
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns).unsqueeze(1)

        # 核心更新循环
        for _ in range(10): # 每批数据重复训练次数
            values = self.critic(states)
            advantage = returns - values.detach() # 优势函数
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) # 标准化优势函数

            # 计算新的 log_probs
            mu, sigma = self.actor(states)
            dist = Normal(mu, sigma)
            new_log_probs = dist.log_prob(actions)
            
            # PPO Ratio: exp(new - old)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO Clip Loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage
            action_loss = -torch.min(surr1, surr2).mean()
            # 在计算 action_loss 时增加：
            entropy = dist.entropy().mean()
            action_loss = action_loss - 0.02 * entropy  # 0.01 是常用系数
            value_loss = F.mse_loss(values, returns)
            
            self.a_opt.zero_grad(); action_loss.backward(); self.a_opt.step()
            self.c_opt.zero_grad(); value_loss.backward(); self.c_opt.step()
        # 在 backward() 之后，step() 之前加入
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

# --- 3. 训练循环 ---
env = gym.make("Pendulum-v1")
ppo = PPO(3, 1)
batch_size = 20
for episode in range(4000):
    state, _ = env.reset()
    mem = {'states':[], 'actions':[], 'log_probs':[], 'rewards':[]}
    ep_reward = 0
    step = 0
    for t in range(200):
        action, log_prob = ppo.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        step +=1
        mem['states'].append(state)
        mem['actions'].append(action)
        mem['log_probs'].append(log_prob)
        mem['rewards'].append((reward + 8) / 8) # 奖励归一化，有助于收敛
        
        state = next_state
        ep_reward += reward
        if terminated or truncated: break
                
    # 关键：当步数达到 batch_size 时才更新
    if step >= batch_size:
        ppo.update(mem)
        # 清空内存
        mem = {'states':[], 'actions':[], 'log_probs':[], 'rewards':[]}
        step = 0
        
    if episode % 10 == 0:
        print(f"Episode: {episode}, Reward: {ep_reward:.2f}, mu: {ppo.mu.item():.2f}, std: {ppo.std.item():.2f}")
# --- 4. 测试 ---
state, _ = env.reset()
# record_video
for _ in range(200):
    action, _ = ppo.select_action(state)
    state, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated: break

env.render()
env.close()