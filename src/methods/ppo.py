import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class PPOBuffer():
    def __init__(self, gamma=0.99, lam=0.95):
        self.buffer = []
        self.state_buf = []
        self.action_buf = []
        self.reward_buf = []
        self.value_buf = []
        self.logp_buf = []
        self.gamma = gamma
        self.lam = lam
        
    def store(self, state, action, reward, value, logp):
        self.state_buf.append(state)
        self.action_buf.append(action)
        self.reward_buf.append(reward)
        self.value_buf.append(value)
        self.logp_buf.append(logp)
    
    def clear(self):
        self.state_buf = []
        self.action_buf = []
        self.reward_buf = []
        self.value_buf = []
        self.logp_buf = []
        
    def reset(self):
        self.buffer = []
        self.clear()
    
    def end_trajectory(self, last_value=0):
        # 1. to np arrays
        rewards = np.array(self.reward_buf + [last_value], dtype=np.float32)
        values = np.array(self.value_buf + [last_value], dtype=np.float32)
        # 2. compute deltas
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        
        # 3. compute advantage estimates (GAE)
        adv = np.zeros_like(deltas, dtype=np.float32)
        running_adv = 0.0
        for t in reversed(range(len(self.action_buf))):
            running_adv = deltas[t] + self.gamma * self.lam * running_adv
            adv[t] = running_adv
        
        returns = np.zeros_like(rewards, dtype=np.float32)
        running_ret = 0.0
        for t in reversed(range(len(rewards))):
            running_ret = rewards[t] + self.gamma * running_ret
            returns[t] = running_ret
        returns = returns[:-1]
        
        # 4. compute returns (Lambda Return)
        # returns = adv + values[:-1]
        
        # 5. store to buffer
        for i in range(len(self.action_buf)):
            self.buffer.append((self.state_buf[i], self.action_buf[i], adv[i], returns[i], self.logp_buf[i]))
        self.clear()

class MLPActor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, hidden_sizes=[64, 64]):
        super(MLPActor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))
        self.action_bound = action_bound

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        return a, self.log_std
    
    def logp_calc(self, state, action):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        # 注意：这里的 action 应该是 raw action (unclamped)
        logp = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return logp, entropy
    
    def sample_action(self, state):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(axis=-1)
        # use tanh to squash the action within [-action_bound, action_bound]
        env_action = torch.tanh(action) * self.action_bound
        # env_action = action.clamp(-self.action_bound, self.action_bound)
        return action, env_action, logp

class MLPCritic(nn.Module):
    def __init__(self, state_dim, hidden_sizes=[64, 64]):
        super(MLPCritic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = nn.Linear(hidden_sizes[1], 1)

    def forward(self, state):
        v = F.relu(self.l1(state))
        v = F.relu(self.l2(v))
        v = self.l3(v)
        return torch.squeeze(v, -1)

class PPO():
    def __init__(self, env, actor_lr=3e-4, critic_lr=1e-3, gamma=0.99, lam=0.95, 
                 eps_clip=0.2, A_epochs=10, C_epochs=80, batch_size=64, 
                 hidden_sizes=[64, 64], target_kl=0.01): # 添加 target_kl
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.action_bound = float(env.action_space.high[0])
        
        self.actor = MLPActor(self.obs_dim, self.act_dim, self.action_bound, hidden_sizes)
        self.critic = MLPCritic(self.obs_dim, hidden_sizes)
        
        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.buffer = PPOBuffer(gamma, lam)
        
        self.eps_clip = eps_clip
        self.A_epochs = A_epochs
        self.C_epochs = C_epochs
        self.batch_size = batch_size
        self.target_kl = target_kl # 存储
    
        # for drawing action
        self.lossV_history = []
        self.lossPi_history = []
        self.kl_history = []
    
    
    def update(self):
        states, actions, adv, returns, old_logprobs = zip(*self.buffer.buffer)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        adv = torch.tensor(np.array(adv), dtype=torch.float32)
        returns = torch.tensor(np.array(returns), dtype=torch.float32)
        logp_old = torch.tensor(np.array(old_logprobs), dtype=torch.float32)
        
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        with torch.no_grad():
            old_total_rewards = returns.sum().item()
            value_pred = self.critic(states)
            # old_value_loss = F.mse_loss(value_pred, returns)
            logp, ent = self.actor.logp_calc(states, actions)
            # old_kl = (logp_old - logp).mean().item()
        
        for i in range(self.A_epochs):
            self.actor.optimizer.zero_grad()
            logp, _ = self.actor.logp_calc(states, actions)
            ratio = torch.exp(logp - logp_old)
            
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            
            kl = (logp_old - logp).mean().item()
            if kl > 1.5 * self.target_kl:
                # print(f"Early stopping at epoch {i} with KL {kl:.4f}")
                break
            actor_loss.backward()
            self.actor.optimizer.step()

        for _ in range(self.C_epochs):
            value_pred = self.critic(states)
            critic_loss = F.mse_loss(value_pred, returns)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
        # Logging (Optional simplified)
        print(f"Update: Policy Loss {actor_loss.item():.4f} | Value Loss {critic_loss.item():.4f} | KL {kl:.4f}")
        self.lossPi_history.append(actor_loss.item())
        self.lossV_history.append(critic_loss.item())
        self.kl_history.append(kl)
        self.buffer.reset()    
        
    def draw_history(self):
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12,4))
        
        plt.subplot(1,3,1)
        plt.plot(self.lossPi_history)
        plt.title("Policy Loss")
        
        plt.subplot(1,3,2)
        plt.plot(self.lossV_history)
        plt.title("Value Loss")
        
        plt.subplot(1,3,3)
        plt.plot(self.kl_history)
        plt.title("KL Divergence")
        
        plt.tight_layout()
        # return plt

if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
    ppo = PPO(env, target_kl=0.02) # 稍微调大一点点 KL 限制，Pendulum 比较耐造
    
    max_epochs = 200 # 总共更新 100 次
    episodes_per_epoch = 20 # 每次更新收集 20 个 episode
    
    total_steps = 0
    # for drawing reward curve
    reward_history = []
    
    
    
    for epoch in range(max_epochs):
        episode_rewards = []
        for _ in range(episodes_per_epoch):
            state, _ = env.reset()
            ep_reward = 0
            while True:
                state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
                
                # 获取动作
                raw_action_tensor, env_action_tensor, logp_tensor = ppo.actor.sample_action(state_tensor)
                
                # 转换: Raw Action 存 buffer, Env Action 给环境
                raw_action = raw_action_tensor.detach().numpy()[0]
                env_action = env_action_tensor.detach().numpy()[0]
                logp = logp_tensor.item()
                
                next_state, reward, terminated, truncated, _ = env.step(env_action)
                
                # 存 Raw Action
                ppo.buffer.store(state, raw_action, reward, ppo.critic(state_tensor).item(), logp)
                
                state = next_state
                ep_reward += reward
                total_steps += 1
                
                if terminated or truncated:
                    last_state_tensor = torch.tensor(next_state, dtype=torch.float32).reshape(1, -1)
                    last_value = ppo.critic(last_state_tensor).item() if truncated else 0
                    ppo.buffer.end_trajectory(last_value)
                    episode_rewards.append(ep_reward)
                    break
        
        avg_reward = np.mean(episode_rewards)
        print(f"Epoch {epoch+1}/{max_epochs} | Avg Reward: {avg_reward:.2f} | Steps: {total_steps}")
        ppo.update()
        
    env.close()
    import matplotlib.pyplot as plt
    ppo.draw_history()
    import os
    os.makedirs("output/picture/", exist_ok=True)
    plt.savefig("output/picture/ppo_training_history.png")
    