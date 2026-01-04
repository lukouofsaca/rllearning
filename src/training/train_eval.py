
# TODO: wrap it as a train process, calling different reward calculation methods.
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
from tqdm import tqdm
from methods import get_method
import argparse

if __name__ == "__main__":
    # argparse for different methods
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--method', type=str, default='ppo', help='RL method to use')
    parser.add_argument('--env', type=str, default='Pendulum-v1', help='Gym environment name')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--record-video', type=bool, default=True, help='Whether to record videos')
    parser.add_argument('--video-dir', type=str, default='output/videos', help='Directory to save videos')
    parser.add_argument('--video-interval', type=int, default=500, help='Interval of episodes to record video')
    parser.add_argument('--checkpoint-dir', type=str, default='output/checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=100, help='Interval of episodes to save checkpoints')
    
    args = parser.parse_args()


    # 训练配置
    env_name = args.env
    env = gym.make(env_name, render_mode=None)

    # 视频录制
    # env = RecordVideo(
    #     env,
    #     video_folder=args.video_dir,
    #     episode_trigger=lambda ep_id: ep_id % args.video_interval == 0,
    #     name_prefix=f"{args.method}_{args.env}"
    # )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = get_method(args.method)(state_dim, action_dim, max_action)

    save_dir = args.checkpoint_dir
    os.makedirs(save_dir, exist_ok=True)

    max_episodes = args.episodes    
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
            agent.store_transition((state, next_state, action, action_logprob, reward, done, value))
            
            state = next_state
            episode_reward += reward

            # 定期更新
            if timestep % update_timestep == 0:
                agent.update()

            if done:
                break
        
        if i_episode % args.checkpoint_interval == 0:
            print(f"Episode {i_episode} \t Reward: {episode_reward:.2f}")
            agent.save(os.path.join(save_dir, "ppo_latest.pt"))

            env.close()