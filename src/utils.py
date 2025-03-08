import numpy as np
import imageio
import torch
import gymnasium
import minigrid
from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper, RGBImgPartialObsWrapper, ReseedWrapper, ActionBonus


def evaluate_agent(model, env, num_episodes=50):
    '''evaluate a policy over multiple episodes.'''
    all_rewards = []
    all_steps = []
    success_count = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        episode_reward, steps = 0, 0
        
        while not (done or truncated):
            action,_,_ = model.get_action(obs)
            obs, reward, done, truncated, _ = env.step(action)
            # check the truncation mechanism
            episode_reward += reward
            steps += 1
        
        if done and not truncated:
            success_count += 1
        
        all_rewards.append(episode_reward)
        all_steps.append(steps)

        metrics = {
            "avg_reward": sum(all_rewards) / num_episodes,
            "success_rate": success_count / num_episodes,
            "avg_steps_to_done": sum(all_steps) / success_count if success_count != 0 else 0,
            "all_rewards": all_rewards,
            "all_steps": all_steps
        }
    return metrics

def record_agent_video(model, env, video_path, fps=10):
    '''Records a single episode of the agent in a MiniGrid environment and saves it as a video'''
    obs, _ = env.reset()
    done = False
    
    with imageio.get_writer(video_path, fps=fps) as video:
        while not done:
            video.append_data(env.render()) 
            action,_,_ = model.get_action(obs)
            obs, _, done, truncated, _ = env.step(action)
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