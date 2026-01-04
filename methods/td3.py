import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
from utils.ReplayBuffer import ReplayBuffer

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)

class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = reward + not_done * gamma * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.it % policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

if __name__ == "__main__":
    # Note: CartPole-v1 is discrete. TD3 is for continuous actions. 
    # Using Pendulum-v1 as a standard continuous environment.
    env = gym.make("Pendulum-v1", render_mode="rgb_array")

    env = RecordVideo(
        env,
        video_folder="videos/",
        episode_trigger=lambda ep_id: ep_id % 100 == 0,  # 每10个episode存一次
        name_prefix="td3_pendulum"
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(state_dim, action_dim, max_action)

    # --- checkpoint / save setup ---
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    SAVE_EVERY = 5000  # save every N training iterations

    def _save(path=None):
        if path is None:
            path = os.path.join(save_dir, f"td3_iter_{agent.it}.pt")
        torch.save({
            'actor': agent.actor.state_dict(),
            'actor_target': agent.actor_target.state_dict(),
            'critic': agent.critic.state_dict(),
            'critic_target': agent.critic_target.state_dict(),
            'actor_optimizer': agent.actor_optimizer.state_dict(),
            'critic_optimizer': agent.critic_optimizer.state_dict(),
            'it': agent.it,
        }, path)

    def _load(path):
        ckpt = torch.load(path, map_location=agent.device)
        agent.actor.load_state_dict(ckpt['actor'])
        agent.actor_target.load_state_dict(ckpt['actor_target'])
        agent.critic.load_state_dict(ckpt['critic'])
        agent.critic_target.load_state_dict(ckpt['critic_target'])
        agent.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        agent.it = ckpt.get('it', agent.it)

    agent.save = _save
    agent.load = _load

    # wrap train to automatically save periodically
    _orig_train = agent.train
    def _train_and_maybe_save(*args, **kwargs):
        _orig_train(*args, **kwargs)
        if agent.it % SAVE_EVERY == 0:
            agent.save(os.path.join(save_dir, f"td3_iter_{agent.it}.pt"))
            # also keep a "latest" copy
            agent.save(os.path.join(save_dir, "td3_latest.pt"))
    agent.train = _train_and_maybe_save
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    state, _ = env.reset()
    for t in tqdm(range(int(1e5))):
        if t < 1000:
            action = env.action_space.sample()
        else:
            action = (agent.select_action(state) + np.random.normal(0, 0.1, size=action_dim)).clip(-max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.add(state, action, next_state, reward, done)
        state = next_state

        if t >= 1000:
            agent.train(replay_buffer)

        if done:

            state, _ = env.reset()
    env.close()