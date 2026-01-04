# an ppo file for RL
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
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        # 使用 tanh 将输出限制在 [-1, 1]，然后缩放到 [ -max_action, max_action ]
        mu = self.mu(x)
        std = torch.exp(self.log_std)
        return mu, std
    def get_dist(self, state):
        mu, std = self.forward(state)
        return Normal(mu, std)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.l3(x)

@register_method("ppo")
class PPO(RLMethod):
    def __init__(self, state_dim, action_dim, max_action, 
                 actor=None,critic=None,
                 params=None,):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if params is None:
            params = {}
        # TODO: use
        self.gamma = params.get("gamma", 0.99)
        self.eps_clip = params.get("eps_clip",0.2)
        self.gae_lambda = params.get("gae_lambda",0.95)
        self.K_epochs = params.get("K_epochs",10)
        self.max_action = max_action
        self.lr = params.get("lr",3e-4)
        if actor is None:
            self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        else:
            self.actor = actor.to(self.device)
        if critic is None:
            self.critic = Critic(state_dim).to(self.device)
        else:
            self.critic = critic.to(self.device)

        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.lr},
            {'params': self.critic.parameters(), 'lr': self.lr}
        ])

        self.buffer = []
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

    def store_transition(self, transition):
        state, next_state, action, action_logprob, reward, done, value = transition
        transition = (state, action, action_logprob, reward, done, value)
        self.buffer.append(transition)

    def update(self):
        self.it += 1
        # 将 buffer 中的数据转换为 tensor
        states = torch.FloatTensor(np.array([t[0] for t in self.buffer])).to(self.device)
        actions = torch.FloatTensor(np.array([t[1] for t in self.buffer])).to(self.device)
        old_logprobs = torch.FloatTensor(np.array([t[2] for t in self.buffer])).to(self.device)
        rewards = [t[3] for t in self.buffer]
        is_terminals = [t[4] for t in self.buffer]
        values = torch.FloatTensor(np.array([t[5] for t in self.buffer])).to(self.device)

        # 计算 Returns 和 GAE (Generalized Advantage Estimation)
        returns = []
        advantages = []
        gae = 0
        with torch.no_grad():
            # 假设最后一个状态后的 value 为 0 (如果是固定步数 rollout)
            next_value = 0 
            for reward, is_terminal, value in zip(reversed(rewards), reversed(is_terminals), reversed(values)):
                mask = 1.0 - is_terminal
                delta = reward + self.gamma * next_value * mask - value
                gae = delta + self.gamma * self.gae_lambda * mask * gae

                advantages.insert(0, gae)
                returns.insert(0, gae + value)

                next_value = value

        returns = torch.FloatTensor(returns).to(self.device).detach()
        advantages = torch.FloatTensor(advantages).to(self.device).detach()
        # 优势函数归一化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 迭代 K 次进行策略和价值网络更新
        for _ in range(self.K_epochs):
            dist = self.actor.get_dist(states)
            logprobs = dist.log_prob(actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            state_values = self.critic(states).squeeze()

            # 计算概率比率 r(theta)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # PPO 裁剪损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # 总损失 = 策略损失 + 价值损失 - 熵奖励
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(state_values, returns)
            entropy_loss = -dist_entropy.mean()

            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 更新完成后清空 buffer
        self.buffer = []

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'it': self.it,
            'type': 'ppo'
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.it = ckpt.get('it', self.it)


if __name__ == "__main__":
    # 训练配置
    env_name = "Pendulum-v1"
    env = gym.make(env_name, render_mode="rgb_array")
    
    # 视频录制
    env = RecordVideo(
        env,
        video_folder="output/videos/",
        episode_trigger=lambda ep_id: ep_id % 100 == 0,
        name_prefix="ppo_pendulum"
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = PPO(state_dim, action_dim, max_action)
    
    save_dir = "output/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    max_episodes = 1000
    update_timestep = 2000  # 每收集 2000 步更新一次
    timestep = 0

    for i_episode in tqdm(range(1, max_episodes + 1)):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(1, 500): # Pendulum 每回合最多 200 步，这里设大一点
            timestep += 1
            
            # 选择动作
            action, action_logprob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储轨迹
            agent.store_transition((state, action, action_logprob, reward, done, value))
            
            state = next_state
            episode_reward += reward

            # 定期更新
            if timestep % update_timestep == 0:
                agent.update()

            if done:
                break
        
        if i_episode % 100 == 0:
            print(f"Episode {i_episode} \t Reward: {episode_reward:.2f}")
            agent.save(os.path.join(save_dir, "ppo_latest.pt"))

    env.close()

if __name__ == "__main__":
    # 训练配置
    env_name = "Pendulum-v1"
    env = gym.make(env_name, render_mode="rgb_array")
    
    # 视频录制
    env = RecordVideo(
        env,
        video_folder="output/videos/",
        episode_trigger=lambda ep_id: ep_id % 100 == 0,
        name_prefix="ppo_pendulum"
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = PPO(state_dim, action_dim, max_action)
    
    save_dir = "output/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    max_episodes = 1000
    update_timestep = 2000  # 每收集 2000 步更新一次
    timestep = 0

    for i_episode in tqdm(range(1, max_episodes + 1)):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(1, 500): # Pendulum 每回合最多 200 步，这里设大一点
            timestep += 1
            
            # 选择动作
            action, action_logprob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储轨迹
            agent.store_transition((state, action, action_logprob, reward, done, value))
            
            state = next_state
            episode_reward += reward

            # 定期更新
            if timestep % update_timestep == 0:
                agent.update()

            if done:
                break
        
        if i_episode % 100 == 0:
            print(f"Episode {i_episode} \t Reward: {episode_reward:.2f}")
            agent.save(os.path.join(save_dir, "ppo_latest.pt"))

    env.close()