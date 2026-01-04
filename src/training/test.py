import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
import argparse
import os
import sys

# 添加 src 目录到路径，以便导入模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from methods import get_method

def load_model(agent, method_name, path):
    if not os.path.exists(path):
        print(f"Error: Model file not found at {path}")
        return

    print(f"Loading model from {path}...")
    checkpoint = torch.load(path, map_location=agent.device)
    
    # 根据不同的保存方式加载
    if isinstance(checkpoint, dict) and 'actor' in checkpoint:
        # 如果保存的是字典包含 actor/critic
        if hasattr(agent, 'actor'):
            agent.actor.load_state_dict(checkpoint['actor'])
        if hasattr(agent, 'critic'):
            agent.critic.load_state_dict(checkpoint['critic'])
        if hasattr(agent, 'q_net'):
            agent.q_net.load_state_dict(checkpoint['q_net'])
    else:
        # 如果直接保存的是 state_dict 或者整个模型
        # 这里假设简单的保存逻辑，视具体 save 实现而定
        try:
            if method_name == 'dqn':
                agent.q_net.load_state_dict(checkpoint)
            elif method_name == 'ppo':
                # 假设 PPO save 保存的是 actor 的 state_dict，或者你需要修改 save 方法
                # 这里尝试加载到 actor
                agent.actor.load_state_dict(checkpoint)
            elif method_name == 'td3':
                agent.actor.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Warning: Direct load failed, trying strict=False or checking keys. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='ppo', help='RL method')
    parser.add_argument('--env', type=str, default='Pendulum-v1', help='Gym environment')
    parser.add_argument('--model-path', type=str, required=True, help='Path to .pt model file')
    parser.add_argument('--output-dir', type=str, default='output/videos', help='Directory to save video')
    parser.add_argument('--episodes', type=int, default=3, help='Number of episodes to record')
    args = parser.parse_args()

    # 1. 创建环境 (使用 render_mode='rgb_array' 以便录制)
    env = gym.make(args.env, render_mode='rgb_array')
    
    # 2. 包装环境以录制视频
    env = RecordVideo(
        env, 
        video_folder=args.output_dir, 
        name_prefix=f"{args.method}_{args.env}",
        episode_trigger=lambda x: True # 录制所有回合
    )

    # 3. 获取环境参数
    state_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        max_action = 1.0
    else:
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

    # 4. 初始化 Agent
    # 注意：这里不需要 params，因为我们只加载权重
    agent = get_method(args.method)(state_dim, action_dim, max_action, params={})
    
    # 5. 加载权重
    load_model(agent, args.method, args.model_path)

    # 6. 设置为评估模式 (针对 DQN 关闭 Epsilon，针对 PPO/TD3 关闭 Dropout 等)
    if args.method == 'dqn':
        agent.epsilon = 0.0
    
    # 7. 运行测试循环
    for ep in range(args.episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # select_action 返回 (action, logprob, value)
            # 在测试时，我们只需要 action
            # 注意：select_action 通常处理 batch，这里输入单个 state
            action, _, _ = agent.select_action(state)
            
            # 如果 action 是 batch 形式 (1, action_dim)，取出来
            if isinstance(action, np.ndarray) and len(action.shape) > 1:
                action = action[0]
            
            # 如果是离散动作但返回了数组，取标量
            if isinstance(env.action_space, gym.spaces.Discrete):
                if isinstance(action, np.ndarray):
                    action = action.item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            state = next_state
            total_reward += reward
            
        print(f"Episode {ep+1} Reward: {total_reward:.2f}")

    env.close()
    print(f"Videos saved to {args.output_dir}")