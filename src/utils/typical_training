
# TODO: wrap it as a train process, calling different reward calculation methods.

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