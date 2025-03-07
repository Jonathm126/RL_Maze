import numpy as np
import imageio
import torch
import gymnasium
import minigrid
from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper, RGBImgPartialObsWrapper, ReseedWrapper, ActionBonus


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
            # use argmax to find the best action
            action = act_dist.probs.argmax().item()
            action_mapped = map_action(action)
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
    steps_to_done = steps_to_done / success_count if success_count != 0 else 0

    return total_rewards, avg_reward, success_rate, steps_to_done

def record_agent_video(device, model, env, video_path, fps=10):
    '''Records a single episode of the agent in a MiniGrid environment and saves it as a video'''
    obs, _ = env.reset()
    torch_obs = preprocess_obs(obs, device)
    done = False
    
    with imageio.get_writer(video_path, fps=fps) as video:
        while not done:
            video.append_data(env.render())  # Store frame for video
            # action_dist, _ = model(torch_obs)  # Agent takes action
            # action = action_dist.sample()
            # action = map_action(action).item()
            action,_,_ = model.get_action(obs)
            obs, _, done, truncated, _ = env.step(action)
            torch_obs = preprocess_obs(obs, device)
            done = done or truncated

    return video_path

def build_envs():
    # env options
    env_names = ["MiniGrid-MultiRoom-N2-S4-v0", "MiniGrid-MultiRoom-N4-S5-v0","MiniGrid-MultiRoom-N6-v0"] 

    # env config
    highlight = False
    render_mode = "rgb_array"

    # build env_0 with two rooms
    env_2_rooms = gymnasium.make(env_names[0], render_mode=render_mode, highlight=highlight)
    env_2_rooms = ImgObsWrapper(RGBImgPartialObsWrapper(env_2_rooms)) 
    
    # build env_01 with 4 rooms
    env_4_rooms = gymnasium.make(env_names[1], render_mode=render_mode, highlight=highlight)
    env_4_rooms = ImgObsWrapper(RGBImgPartialObsWrapper(env_4_rooms)) 
    
    # next env
    env_6_rooms = gymnasium.make(env_names[2], render_mode=render_mode, highlight=highlight)
    env_6_rooms = ImgObsWrapper(RGBImgPartialObsWrapper(env_6_rooms)) 
    
    return env_2_rooms, env_4_rooms, env_6_rooms

def preprocess_obs(obs, device):
    ''' preprocess an observation to torch image and cast to float. '''
    np_obs = torch.from_numpy(obs).type(torch.float32)
    np_obs = np_obs / 255.0 # scale to 0-1
    np_obs = np_obs.permute(2, 0, 1).unsqueeze(0).to(device)
    return np_obs

def map_action(action):
    ''' maps a 0-3 action space to 0-6, with 3, 4, 6 unused.'''
    # if tensor
    if isinstance(action, torch.Tensor):
        return torch.where(action == 3, torch.tensor(5), action)
    else:
        # if not a tensor
        return 5 if action == 3 else action