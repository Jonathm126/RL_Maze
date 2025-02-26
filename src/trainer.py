import torch
import torch.distributions 
import torch.nn.functional as F
from dataclasses import dataclass

# my imports
from src.format_utils import preproess_obs, map_action

@dataclass
class Trainer():
    def __init__(self, device, model, writer, model_path = None):
        self.device = device
        self.episode = 0
        self.model = model
        self.writer = writer
        self.path = model_path
    
    def reset(self, model):
        self.model = model
        self.episode = 0
        self.step = 0
    
    def A2CTrain(self, env, optimizer, l_params, max_episodes, max_attempts = None):
        # unpack
        start_episode = self.episode
        end_episode = max_episodes + start_episode
        attempts = 0
        
        # set model
        self.optimizer = optimizer(self.model.parameters(), lr=l_params['lr'])
        self.model.train()
        
        # iterate on eposides
        for self.episode in range (start_episode, end_episode):
            # reset environment if first episode or if max attempts exceeded or done
            # TODO deal with truncation mechanism
            # if done or attempts >= max_attempts:
            #     attempts = 0 
            # else:
            #     # Continue from the current state without resetting the environment
            #     obs = torch_obs 
                
            # get initial observation
            obs, _ = env.reset()
            torch_obs = preproess_obs(obs, self.device)
            
            # reset params
            done, truncated = False, False
            episode_return = 0
            
            # preform steps
            while not done and not truncated:
                # get model outputs
                act_dist, value = self.model(torch_obs)
                action = act_dist.sample()
                log_prob_action = act_dist.log_prob(action)
                
                # map action 3 to 5 (unused actions)
                action_mapped = map_action(action)
                
                # preform the action
                next_obs, reward, done, truncated ,_ = env.step(action_mapped)
                torch_next_obs = preproess_obs(next_obs, self.device)
                
                # use the critic to estimate the value of the next step
                with torch.no_grad():
                    _, next_value = self.model(torch_next_obs)
                
                # critic loss loss
                td_target = reward + l_params['gamma'] * next_value * (1 - done)
                advantage = td_target - value
                critic_loss = F.mse_loss(value, td_target.detach())
                
                # actor loss
                actor_loss = -log_prob_action * advantage.detach()
                
                # total loss
                loss = l_params['critic_lr_weight'] * critic_loss + \
                        l_params['actor_lr_weight'] * actor_loss
                
                # preform optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # update params
                torch_obs = torch_next_obs
                episode_return += reward
                attempts += 1
            
            # Record statistics
            self.writer.add_scalar('Loss/Actor', actor_loss.item(), self.episode)
            self.writer.add_scalar('Loss/Critic', critic_loss.item(), self.episode)
            self.writer.add_scalar('Returns/Episode Return', episode_return, self.episode)