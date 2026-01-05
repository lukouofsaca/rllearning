# an ppo file for RL
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from methods.registry import register_method
from methods.abstract_methods import RLMethod
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from tqdm import tqdm
import os

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(-5, 2)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self(state)
        std = log_std.exp()
        dist = Normal(mu, std)

        z = dist.rsample()                # reparameterization
        action = torch.tanh(z)

        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(-1)

        return action, log_prob

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
        self.gae_lambda = params.get("gae_lambda", 0.97)
        
        self.C_epochs = params.get("C_epochs",80)
        self.A_epochs = params.get("A_epochs", 80)
        self.lr = params.get("lr", 3e-4)
        self.max_action = max_action
        self.target_kl = params.get("target_kl", 0.01)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.actor.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': 3e-4},
        ])
        self.critic.optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.buffer = []
        self.it = 0
        self.kl_history = []

    # def select_action(self, state):
        # state: (num_envs, state_dim) or (state_dim,)
        # if isinstance(state, tuple):
        #     state = state[0]
        # state = torch.FloatTensor(state).to(self.device)
        # state = state.reshape(1, -1)
        # with torch.no_grad():
        #     dist = self.actor.get_dist(state)
        #     action = dist.sample()
        #     action_logprob = dist.log_prob(action).sum(dim=-1)
        #     value = self.critic(state)
        
        # action = self.max_action * F.tanh(action)
        # return action.cpu().data.numpy().flatten(), action_logprob.cpu().data.numpy().flatten(), value.cpu().data.numpy().flatten()
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, logprob = self.actor.sample(state)
            value = self.critic(state)

        action = action.cpu().numpy()[0] * self.max_action
        return action, logprob.item(), value.item()


    def store_transition(self, transition):
        """
        transition 内容: (state, next_state, action, action_logprob, reward, done, value)
        """
        state, next_state, action, action_logprob, reward, done, value = transition

        # 1. 强制将状态拍扁为一维 (state_dim,)
        state_fixed = np.array(state).flatten()
        next_state_fixed = np.array(next_state).flatten()
        # 2. 强制将动作拍扁 (action_dim,)
        action_fixed = np.array(action).flatten()
        # 3. 将标量数据转为纯 Python 标量，防止带入 (1,) 形状的数组
        # 尤其是 reward 和 value，经常会有 (1,) 和 标量的混用
        action_logprob_fixed = float(np.array(action_logprob).item())
        reward_fixed = float(np.array(reward).item())
        done_fixed = float(np.array(done).item())
        value_fixed = float(np.array(value).item())
        # 重新打包存入 buffer
        fixed_transition = (
            state_fixed, 
            next_state_fixed, 
            action_fixed, 
            action_logprob_fixed, 
            reward_fixed, 
            done_fixed, 
            value_fixed
        )
        self.buffer.append(fixed_transition)
        self.dist = (0,1)  # clear cached dist

    def update(self):
        
        states, next_states, actions, old_logprobs, rewards, dones, values = zip(*self.buffer)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        old_logprobs = torch.tensor(old_logprobs, dtype=torch.float32).to(self.device)

        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)
        # bootstrap
        with torch.no_grad():
            last_val = 0 if dones[-1] else self.critic(states[-1]).item()
        values = np.append(values, last_val)

        # GAE
        adv = np.zeros(len(rewards))
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1-dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1-dones[t]) * gae
            adv[t] = gae

        returns = adv + values[:-1]

        adv = torch.tensor(adv, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # actor update
        print("start update")
        for i in range(self.A_epochs):
            new_actions, logprobs = self.actor.sample(states)
            ratio = (logprobs - old_logprobs).exp()

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            self.actor.optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor.optimizer.step()
            # use kl for early stopping (optional)
            with torch.no_grad():
                kl = (old_logprobs - logprobs).mean()
            if kl > 1.5 * self.target_kl:
                print(f"Early stopping at step {i} due to reaching max KL {kl:.6f}")
                break
            if i == self.A_epochs - 1:
                print(f"Final KL after actor update: {kl.item():.6f}")
            self.kl_history.append(kl.item())


        # critic update
        for i in range(self.C_epochs):
            value_loss = F.mse_loss(self.critic(states).squeeze(), returns)

            self.critic.optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic.optimizer.step()
        self.it += 1

        self.buffer.clear()
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

    max_episodes = 10000
    update_episode = 20
    timestep = 0
    episode_rewards = []
    avg_episode_rewards = []
    episode_reward = 0
    for i_episode in tqdm(range(1, max_episodes + 1)):
        state, _ = env.reset()
        episode_reward = 0
        
        for t in range(1, 400): # Pendulum 每回合最多 200 步，这里设大一点
            timestep += 1
            
            # 选择动作
            action, action_logprob, value = agent.select_action(state)
            # action = 2 * F.tanh(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储轨迹
            # if(state.shape!=(3,1)):
            #     print(state.shape,timestep)
            # if(next_state.shape!=(3,1)):
            #     print(next_state.shape,timestep)
            agent.store_transition((state, next_state, action, action_logprob, reward, done, value))
            
            state = next_state
            episode_reward += reward
              

            if done:
                break
        episode_rewards.append(episode_reward)
        if i_episode % update_episode == 0:
            avg_episode_reward = sum(episode_rewards[-update_episode:]) / update_episode
            avg_episode_rewards.append(avg_episode_reward)
            print(f"Episode {i_episode} \t Average Reward: {avg_episode_reward:.2f}")
            agent.update()      

        if i_episode % 100 == 0:            
            agent.save(os.path.join(save_dir, "ppo_latest.pt"))
    env.close()
    # save rewards
    report_dir = "output/reports"
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "rewards.txt"), "w") as f:
        for r in episode_rewards:
            f.write(f"{r}\n")
    with open(os.path.join(report_dir, "kl.txt"), "w") as f:
        for k in agent.kl_history:
            f.write(f"{k}\n")
    import matplotlib.pyplot as plt
    plt.plot(avg_episode_rewards)
    plt.xlabel(f"Episode (per {update_episode} episodes)")
    plt.ylabel("avg Reward")
    plt.savefig(os.path.join(report_dir, "rewards.png"))
