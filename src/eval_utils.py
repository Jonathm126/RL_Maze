import numpy as np
import imageio
from src.format_utils import preprocess_obs, map_action

def evaluate_agent_rewards(device, model, env, num_episodes=10, max_steps = None):
    '''evaluate a policy over multiple episodes.'''
    total_rewards = []
    success_count = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        torch_obs = preprocess_obs(obs, device)
        done, truncated = False, False
        episode_reward, steps, steps_to_done = 0, 0, 0 

        while not (done or truncated):
            # select action
            act_dist, _ = model(torch_obs)
            action = act_dist.sample()
            action_mapped = map_action(action).item()
            # preform step
            obs, reward, done, truncated, _ = env.step(action_mapped)
            torch_obs = preprocess_obs(obs, device)
            # check the truncation mechanism
            if max_steps is not None: 
            # if we use the max steps mechanism ignore the truncation
                truncated = False
            # count
            episode_reward += reward
            steps += 1
            # if the max_steps mechanism is used
            if max_steps is not None:
                if steps >= max_steps:
                    break

        total_rewards.append(episode_reward)
        if done: 
            success_count += 1
            steps_to_done += steps

    avg_reward = np.mean(total_rewards)
    success_rate = success_count / num_episodes
    steps_to_done = steps_to_done / success_count # normalize by number of sucesses

    return total_rewards, avg_reward, success_rate, steps_to_done

def record_agent_video(device, model, env, video_path, fps=10):
    '''Records a single episode of the agent in a MiniGrid environment and saves it as a video'''
    obs, _ = env.reset()
    torch_obs = preprocess_obs(obs, device)
    done = False
    
    with imageio.get_writer(video_path, fps=fps) as video:
        while not done:
            video.append_data(env.render())  # Store frame for video
            action_dist, _ = model(torch_obs)  # Agent takes action
            action = action_dist.sample()
            action = map_action(action).item()
            obs, _, done, truncated, _ = env.step(action)
            torch_obs = preprocess_obs(obs, device)
            done = done or truncated

    return video_path
