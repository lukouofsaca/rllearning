import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

from methods.abstract_methods import RLMethod
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_sizes=[256, 256]):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.mu = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std = nn.Parameter(-0.5*torch.zeros(action_dim))    
        self.max_action = max_action
    
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mu = self.mu(a)
        log_std = self.log_std
        return mu, log_std
    
    def sample_action(self, state):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        action = dist.rsample()  # reparameterization trick
        logp = dist.log_prob(action).sum(axis=-1)
        
        # Apply squashing function
        env_action = torch.tanh(action) * self.max_action
        # Another way to compute log prob after squashing
        return action, env_action, logp
    
    def logp_calc(self, state, action):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        # 注意：这里的 action 应该是 raw action (unclamped)
        logp = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        return logp, entropy
    
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=[64,64],set_type='Q'):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.q = nn.Linear(hidden_sizes[1], 1)
        self.Ctype = set_type  # to distinguish Q and V networks
    
    def forward(self, state, action):
        if self.Ctype != 'V':
            x = torch.cat([state, action], dim=-1)
        else:
            x = state
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        q_value = self.q(x)
        return q_value

class SACReplayBuffer():
    # Note:
    # The SACBuffer stores 
    # ONLY THINGS ABOUT ENV. 
    # NOTHING RELATED WITH AGENT 
    # would be stored here.
    # TODO: put all the buffers into GPU memory
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.terminated = np.zeros((max_size, 1),dtype=np.bool_)
        self.truncated = np.zeros((max_size, 1),dtype=np.bool_)
        
    def store_transition(self, transition):
        state, action, next_state, reward, terminated, truncated = transition
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.terminated[self.ptr] = terminated
        self.truncated[self.ptr] = truncated
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def end_trajectory(self):
        # sac does not need to do anything at the end of trajectory
        None
        pass
        
    def sample_batch(self, batch_size):
        for _ in range(10):  # try 10 times to sample a valid batch
            idxs = np.random.randint(0, self.size, size=batch_size)
            batch = dict(
                state = torch.FloatTensor(self.state[idxs]),
                action = torch.FloatTensor(self.action[idxs]),
                next_state = torch.FloatTensor(self.next_state[idxs]),
                reward = torch.FloatTensor(self.reward[idxs]),
                truncated = torch.FloatTensor(self.truncated[idxs]),
                terminated = torch.FloatTensor(self.terminated[idxs]),
            )
            return batch
    def clear(self):
        self.ptr = 0
        self.size = 0
        

class SAC():
    """
    SAC: Soft Actor-Critic
    Update Goal:
    Maximum Q + entropy
    J = E[Q + lambda * H(pi(.|s)) ]
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        params=None
    ):
      
      
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize actor, critic, target critic, optimizers, replay buffer, etc.
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        # double Q networks
        self.qcritic1 = Critic(state_dim, action_dim).to(self.device)
        self.qcritic2 = Critic(state_dim, action_dim).to(self.device)
        # V networks
        self.Vcritic = Critic(state_dim, 0, set_type='V').to(self.device)
        self.Vcritic_target = Critic(state_dim, 0, set_type='V').to(self.device)
        
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params.get("actor_lr", 3e-4))
        self.qcritic1_optimizer = optim.Adam(self.qcritic1.parameters(), lr=params.get("qcritic_lr", 1e-4))
        self.qcritic2_optimizer = optim.Adam(self.qcritic2.parameters(), lr=params.get("critic_lr", 1e-4))
        self.Vcritic.optimizer = optim.Adam(self.Vcritic.parameters(), lr=params.get("critic_lr", 1e-3))
        self.Vcritic_target.optimizer = optim.Adam(self.Vcritic_target.parameters(), lr=params.get("critic_lr", 1e-3))
        
        
        self.lambdav = params.get("lambda", 0.2)                # entropy coefficient. equals to alpha in original paper. [[[ V --target--> E(Q - alpha * logpi(a|s))  ]]]
        self.lambdaq = params.get("lambda_q", 0.5)              # Q-value loss coefficient
        self.gamma = params.get("gamma", 0.99)                  # discount factor
        self.tau = params.get("tau", 0.005)                     # target network update rate
        
        self.buffer = SACReplayBuffer(params.get("buffer_size", int(1e6)), state_dim, action_dim)
        self.batch_size = params.get("batch_size", 256)
        self.Q_update_steps = params.get("Q_update_steps", 1)
        self.V_update_steps = params.get("V_update_steps", 1)        
        
    def Qupdate(self, network, optimizer, states, actions, rewards, next_states, q_target, teminated):
        # Update Q networks
        q_pred = network(states, actions)
        q_loss = F.mse_loss(q_pred, q_target).mean()
        # Optimize Q networks
        optimizer.zero_grad()
        q_loss.backward()
        optimizer.step()
        
        return q_loss.item()
    
    def soft_update(self, network, target_network, tau):
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def Vupdate(self, states):
        # Update V network
    
        # 2.1. V_goal = E_{a~pi}[ min(Q1(s,a), Q2(s,a)) - lambda * logpi(a|s) ]
        with torch.no_grad():
            action, _, logp = self.actor.sample_action(states)
            q1_value = self.qcritic1(states, action)
            q2_value = self.qcritic2(states, action)
            min_q = torch.min(q1_value, q2_value)
            ent = self.lambdav * logp.unsqueeze(-1)
            v_target = min_q - ent
        
        # 2.2. V_loss = MSE( V(s), V_goal )
        v_pred = self.Vcritic(states, None)
        v_loss = F.mse_loss(v_pred, v_target).mean()
        
        # 2.3. Optimize V network
        self.Vcritic.optimizer.zero_grad()
        v_loss.backward()
        self.Vcritic.optimizer.step()
        
        # 2.4. Soft update V target network
        self.soft_update(self.Vcritic, self.Vcritic_target, self.tau)
        return v_loss.item(), ent.mean().item()
    
    def Actor_update(self, states):
        # 3. Update Actor network
        action, _, logp = self.actor.sample_action(states)
        q1_value = self.qcritic1(states, action)
        q2_value = self.qcritic2(states, action)
        min_q = torch.min(q1_value, q2_value)
        
        # 3.1. Actor_loss = E_{s~D}[ lambda * logpi(a|s) - min(Q1(s,a), Q2(s,a)) ]
        actor_loss = (self.lambdav * logp.unsqueeze(-1) - min_q).mean()
        
        # 3.2. Optimize Actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item()
    
    def update(self):
        # Data Process
        batch = self.buffer.sample_batch(self.batch_size)
        states = batch['state'].to(self.device)
        actions = batch['action'].to(self.device)
        next_states = batch['next_state'].to(self.device)
        rewards = batch['reward'].to(self.device)
        terminated = batch['terminated'].to(self.device)
        truncated = batch['truncated'].to(self.device)
        dones = torch.max(terminated, truncated)
        
        # 1. Update Q networks
        # Qt = R + gamma * (1 - terminated) * Vtarget(s')
        # 需要把Vcritec_target的输出detach掉，避免梯度传递到V网络
        with torch.no_grad():
            q_target = rewards + self.gamma * (1 - terminated) * self.Vcritic_target(next_states, None)
        # Target Q value(for t times)
        for i in range(self.Q_update_steps):
            lossq1 = self.Qupdate(self.qcritic1, self.qcritic1_optimizer, states, actions, rewards, next_states, q_target, terminated)
            lossq2 = self.Qupdate(self.qcritic2, self.qcritic2_optimizer, states, actions, rewards, next_states, q_target, terminated)
        
        # 2. Update V network
        for i in range(self.V_update_steps):
            lossv, ent = self.Vupdate(states)
        
        # 3. Update Actor network
        lossa = self.Actor_update(states)
        
        # Logging (Optional simplified)
        # print(f"SAC Update: LossQ1 {lossq1:.4f} | LossQ2 {lossq2:.4f} | LossV {lossv:.4f} | LossA {lossa:.4f} | Entropy {ent:.4f}")
        return lossq1, lossq2, lossv, lossa, ent





def experience_sample(sac,env,state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(sac.device)
    action, env_action, _ = sac.actor.sample_action(state_tensor)
    
    env_action = env_action.detach().cpu().numpy()[0]
    action = action.detach().cpu().numpy()[0]
    
    next_state, reward, terminated, truncated, _ = env.step(env_action)
    sac.buffer.store_transition((state, action, next_state, reward, terminated, truncated))

def main():
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    sac_params = {
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "buffer_size": int(1e6),
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.02,
        "lambda": 0.2,
        "Q_update_steps": 3,
        "V_update_steps": 1,
    }
    
    sac = SAC(state_dim, action_dim, max_action, sac_params)
    
    num_episodes = 100
    max_steps = 200
    reward_history = []
    reward_sum = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        reward_sum = 0
        for step in range(max_steps):
            
            experience_sample(sac, env, state)
            sac.update()
            state, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            reward_sum += reward
            if terminated or truncated:
                break
        reward_history.append(reward_sum)
        print(f"Episode {episode+1}, Reward: {reward_sum}")
        reward_sum = 0
    
    env.close()
if __name__ == "__main__":
    main()
