import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import os
from tqdm import tqdm

# --- Trick 1: 网络正交初始化 (Orthogonal Initialization) ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=64):
        super().__init__()
        self.l1 = layer_init(nn.Linear(state_dim, hidden_dim))
        self.l2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        # 最后一层输出 mu，缩放设小 (0.01) 以确保初始动作接近中值
        self.mu = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        # 将 log_std 设为独立参数，初值 -0.5 (std≈0.6)
        self.log_std = nn.Parameter(torch.full((1, action_dim), -0.5))
        self.max_action = max_action

    def forward(self, state):
        x = torch.tanh(self.l1(state))
        x = torch.tanh(self.l2(x))
        mu = self.max_action * torch.tanh(self.mu(x))
        std = torch.exp(self.log_std) # std 也会随训练自动调整
        return mu, std

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.l1 = layer_init(nn.Linear(state_dim, hidden_dim))
        self.l2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.l3 = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, state):
        x = torch.tanh(self.l1(state))
        x = torch.tanh(self.l2(x))
        return self.l3(x)

class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.eps_clip = 0.2
        self.K_epochs = 10
        self.batch_size = 64
        
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        # 学习率建议：Actor 略小，Critic 略大，加入 eps 提升数值稳定性
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': 3e-4, 'eps': 1e-5},
            {'params': self.critic.parameters(), 'lr': 1e-3, 'eps': 1e-5}
        ])

        self.buffer = []

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mu, std = self.actor(state)
            dist = Normal(mu, std)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(state)
        return action.cpu().numpy().flatten(), action_logprob.cpu().numpy().flatten(), value.cpu().numpy().flatten()
    def store_transition(self, transition):
            # transition 是一个元组: (s, a, log_p, r, done, v)
            self.buffer.append(transition)
    def update(self):
        # 提取数据并强制展平为一维 (2000,)
        s = torch.FloatTensor(np.array([t[0] for t in self.buffer])).to(self.device)
        a = torch.FloatTensor(np.array([t[1] for t in self.buffer])).to(self.device)
        
        # 关键修改：所有的一维数值张量都加上 .view(-1)
        old_lp = torch.FloatTensor(np.array([t[2] for t in self.buffer])).to(self.device).view(-1)
        r = torch.FloatTensor(np.array([t[3] for t in self.buffer])).to(self.device).view(-1)
        dw = torch.FloatTensor(np.array([t[4] for t in self.buffer])).to(self.device).view(-1)
        v = torch.FloatTensor(np.array([t[5] for t in self.buffer])).to(self.device).view(-1)

        # --- Trick 2: GAE (Generalized Advantage Estimation) ---
        adv = torch.zeros_like(r).to(self.device)
        gae = 0
        with torch.no_grad():
            next_v = 0 # 也可以用 critic(next_state) 估计，但此处为简化处理
            for t in reversed(range(len(r))):
                delta = r[t] + self.gamma * next_v * (1 - dw[t]) - v[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dw[t]) * gae
                adv[t] = gae
                next_v = v[t]
            ret = adv + v

        # --- Trick 3: 优势函数归一化 ---
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # --- Trick 4: Mini-batch 更新 ---
        idx = np.arange(len(self.buffer))
        for _ in range(self.K_epochs):
            np.random.shuffle(idx)
            for start in range(0, len(self.buffer), self.batch_size):
                b_idx = idx[start : start + self.batch_size]
                
                mu, std = self.actor(s[b_idx])
                dist = Normal(mu, std)
                new_lp = dist.log_prob(a[b_idx]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                new_v = self.critic(s[b_idx]).squeeze()

                ratio = torch.exp(new_lp - old_lp[b_idx])
                
                # PPO 裁剪损失
                surr1 = ratio * adv[b_idx]
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv[b_idx]
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_v, ret[b_idx])
                # 组合 Loss，设置熵系数 0.01 鼓励探索
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                # --- Trick 5: 梯度裁剪 (Gradient Clipping) ---
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()

        self.buffer = []

if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], float(env.action_space.high[0]))
    
    total_steps = 0
    update_interval = 8000 # 收集 2000 步更新一次
    
    for i_ep in range(8000):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(200):
            total_steps += 1
            action, lp, v = agent.select_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            
            # --- Trick 6: 奖励缩放 (Reward Scaling) ---
            # Pendulum 原始奖励约为 -16.0 到 0.0，除以 10 帮助模型稳定收敛
            agent.store_transition((state, action, lp, reward / 10.0, term or trunc, v))
            
            state = next_state
            ep_reward += reward
            
            if total_steps % update_interval == 0:
                agent.update()
            
            if term or trunc: break
            
        if i_ep % 20 == 0:
            print(f"Ep: {i_ep} \t Reward: {ep_reward:.2f} \t Std: {torch.exp(agent.actor.log_std).detach().cpu().numpy().flatten()}")