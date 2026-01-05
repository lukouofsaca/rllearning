# an ppo file for RL
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from methods.registry import register_method
from methods.abstract_methods import RLMethod

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mu = self.max_action * torch.tanh(self.mu(x))
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
    def __init__(self, state_dim, action_dim, max_action, params=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if params is None:
            params = {}
        self.params = params
        self.gamma = params.get("gamma", 0.99)
        self.eps_clip = params.get("eps_clip", 0.2)
        self.gae_lambda = params.get("gae_lambda", 0.95)
        self.K_epochs = params.get("K_epochs", 10)
        self.lr = params.get("lr", 3e-4)
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.lr},
            {'params': self.critic.parameters(), 'lr': self.lr}
        ])

        self.buffer = []
        self.it = 0

    def select_action(self, state):
        # state: (num_envs, state_dim) or (state_dim,)
        if isinstance(state, tuple):
            state = state[0]
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            dist = self.actor.get_dist(state)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(state)
        
        action = torch.clamp(action, -self.max_action, self.max_action)
        return action.cpu().data.numpy(), action_logprob.cpu().data.numpy(), value.cpu().data.numpy()

    def store_transition(self, transition):
        # transition: (state, next_state, action, action_logprob, reward, done, value)
        # We store the batch directly.
        self.buffer.append(transition)

    def update(self):
        if len(self.buffer) == 0:
            return

        # Optimize buffer processing: Transpose list of tuples to tuple of lists
        batch = list(zip(*self.buffer))
        
        # Convert to tensors efficiently
        # Stack along time dimension (dim 0)
        # Result shape: (buffer_steps, num_envs, ...)
        states = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(self.device)
        # next_states = torch.tensor(np.array(batch[1]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(batch[2]), dtype=torch.float32).to(self.device)
        old_logprobs = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(batch[4]), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(batch[5]), dtype=torch.float32).to(self.device)
        values = torch.tensor(np.array(batch[6]), dtype=torch.float32).to(self.device)

        # Calculate Advantages using GAE
        with torch.no_grad():
            # We need next states for all steps to compute next_values
            # But actually we only need next_values for the GAE calculation.
            # If we stored next_states, we can use them.
            next_states = torch.tensor(np.array(batch[1]), dtype=torch.float32).to(self.device)
            next_values = self.critic(next_states).squeeze(-1) # (buffer_steps, num_envs)
            
            # Values shape: (buffer_steps, num_envs) (if critic output is (batch, 1))
            if values.dim() == 3: values = values.squeeze(-1)
            
            # delta = r + gamma * V(s') * (1-done) - V(s)
            deltas = rewards + self.gamma * next_values * (1 - dones) - values
            
            # Vectorized GAE calculation
            deltas_np = deltas.cpu().numpy()
            dones_np = dones.cpu().numpy()
            advantages_np = np.zeros_like(deltas_np)
            gae = 0
            
            # Loop backwards over time steps
            for t in reversed(range(len(deltas_np))):
                # If done, reset GAE. dones_np[t] is (num_envs,)
                # We need to handle masking per environment.
                # gae = delta + gamma * lambda * gae * (1 - done)
                gae = deltas_np[t] + self.gamma * self.gae_lambda * gae * (1 - dones_np[t])
                advantages_np[t] = gae
                
            advantages = torch.tensor(advantages_np, dtype=torch.float32).to(self.device)
        
        returns = advantages + values

        # Flatten the batch for training: (buffer_steps * num_envs, ...)
        states = states.reshape(-1, states.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        old_logprobs = old_logprobs.reshape(-1)
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Calculate current logprobs and values
            dist = self.actor.get_dist(states)
            logprobs = dist.log_prob(actions).sum(dim=-1)
            dist_entropy = dist.entropy().sum(dim=-1)
            state_values = self.critic(states).squeeze()

            # Ratio
            ratios = torch.exp(logprobs - old_logprobs)

            # Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(state_values, returns) - 0.01 * dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.buffer = []
        self.it += 1

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
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