import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

# --- 1. 网络定义 ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.net(state)

# --- 2. PPO 智能体 ---
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.gamma = 0.99
        self.lam = 0.95
        self.eps_clip = 0.2
        self.k_epochs = 10
        self.batch_size = 64
        self.entropy_coef = 0.01

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def update(self, memory):
        # 将列表转换为 numpy 数组再转 tensor，速度更快且能处理形状
        states = torch.FloatTensor(np.array(memory['states']))
        actions = torch.LongTensor(np.array(memory['actions']))
        old_logprobs = torch.FloatTensor(np.array(memory['logprobs']))
        rewards = memory['rewards']
        is_terminals = memory['is_terminals']

        # --- 步骤 A: 计算 GAE 和 Returns ---
        returns = []
        advantages = []
        
        with torch.no_grad():
            # 【修复点 1】: 使用 squeeze(-1) 确保只压缩最后一维，或者用 view(-1)
            # 这样即使只有 1 个样本，shape 也会是 (1,) 而不是 ()
            values = self.critic(states).view(-1).cpu().numpy()
            
            gae = 0
            next_value = 0
            for i in reversed(range(len(rewards))):
                mask = 1.0 - is_terminals[i]
                # 这里 values[i] 现在安全了
                delta = rewards[i] + self.gamma * next_value * mask - values[i]
                gae = delta + self.gamma * self.lam * mask * gae
                advantages.insert(0, gae)
                next_value = values[i]
                returns.insert(0, gae + values[i])

        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # 优势归一化
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 步骤 B: 迭代更新 K 次 ---
        for _ in range(self.k_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                idx = indices[start : start + self.batch_size]
                
                # 采样当前批次
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_logprobs = old_logprobs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # 1. 更新 Actor (Policy Loss)
                probs = self.actor(batch_states)
                dist = torch.distributions.Categorical(probs)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_opt.step()

                # 2. 更新 Critic (Value Loss)
                curr_values = self.critic(batch_states).view(-1)
                critic_loss = nn.MSELoss()(curr_values, batch_returns)
                
                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_opt.step()

# --- 3. 训练主程序 ---
def train():
    env = gym.make('CartPole-v1')
    agent = PPOAgent(state_dim=4, action_dim=2)
    
    max_episodes = 20000
    update_timestep = 500  # 每收集500步更新一次
    timestep = 0

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}
        
        for t in range(1, 501): # CartPole 最大500步
            timestep += 1
            action, logprob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory['states'].append(state)
            memory['actions'].append(action)
            memory['logprobs'].append(logprob)
            memory['rewards'].append(reward)
            memory['is_terminals'].append(done)

            state = next_state
            episode_reward += reward

            if timestep % update_timestep == 0:
                agent.update(memory)
                memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}

            if done:
                break
        
        if episode % 20 == 0:
            print(f"Episode {episode} \t Reward: {episode_reward}")
        
        if episode_reward >= 490: # CartPole-v1 胜利条件
            print(f"Solved! Final Reward: {episode_reward}")
            break

    env.close()

if __name__ == "__main__":
    train()