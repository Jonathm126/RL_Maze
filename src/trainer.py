import torch
import torch.distributions 
import torch.nn.functional as F

# my imports
from src.format_utils import preprocess_obs, map_action
from src.eval_utils import evaluate_agent_rewards
from src.train_utils import RolloutBuffer

class Trainer():
    def __init__(self, device, model, optimizer, writer, model_path = None):
        self.device = device
        self.episode = 0
        self.model = model
        self.optimizer = optimizer
        self.writer = writer
        self.path = model_path
        self.experiment_phase = None
    
    def reset_episodes(self):
        self.episode = 0
    
    def set_phase(self, phase):
        self.experiment_phase = phase
    
    def set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    def A2Closs(self, l_params, rollout:RolloutBuffer, next_value):
        '''compute a2c losses for tensors of values'''
        # rollout size
        n_rollout = rollout.position
        gamma = l_params['gamma']
        
        # compute returns
        returns = torch.cat([torch.zeros(n_rollout, device=self.device), 
                            next_value], dim=-1)
        for t in reversed(range(n_rollout)):
            returns[t] = rollout['rewards'][t] + gamma * returns[t + 1] * (1 - rollout['dones'][t]) 
        returns = returns[:-1]
        
        # compute the advantage function
        advantage = returns - rollout['values'][0:n_rollout]
        # value loss (critic)
        critic_loss = F.mse_loss(rollout['values'][0:n_rollout], returns.detach(), reduction='mean')
        # actor loss 
        actor_loss = -(rollout['log_probs'][0:n_rollout] * advantage.detach()).mean()
        # entropy loss
        entropy_loss = rollout['entropies'][0:n_rollout].mean()
        
        return actor_loss, critic_loss, entropy_loss
    
    def A2CTrain(self, env, l_params, episodes, val_every_episodes, rollout_size, episode_max_steps, save_every_episodes = 1000):
        '''Traines a A2C Policy'''
        start_episode = self.episode
        end_episode = episodes + start_episode
        
        # set model and buffer
        self.model.train()
        self.buffer = RolloutBuffer(self.device, rollout_size)
        
        # episodes loop
        for self.episode in range (start_episode, end_episode):
            # get initial observation
            obs, _ = env.reset()
            torch_obs = preprocess_obs(obs, self.device)
            
            # reset params
            done = False
            episode_return, ep_step = 0, 0
            
            # steps inside episode loop
            while not done and ep_step <= episode_max_steps:
                # reset params and set buffer_full=False
                buffer_full = self.buffer.reset()
                
                # rollout loop
                while not buffer_full and not done and ep_step <= episode_max_steps :
                    # get model outputs
                    act_dist, value = self.model(torch_obs)
                    
                    # compute action and params
                    action = act_dist.sample()
                    log_prob_action = act_dist.log_prob(action)
                    action_entropy = act_dist.entropy()
                    
                    # map action 3 to 5 (unused actions)
                    action_mapped = map_action(action).item()
                    self.writer.add_scalar(f'{self.experiment_phase}/Action Taken', action_mapped, self.episode)
                    
                    # preform step
                    next_obs, reward, done, truncated, _ = env.step(action_mapped)
                    torch_next_obs = preprocess_obs(next_obs, self.device)
                    reward = reward * l_params['reward_amplify']
                    
                    # append to rollout buffer
                    buffer_full = self.buffer.add(log_prob_action, value, reward, done, action_entropy)
                    
                    torch_obs = torch_next_obs
                    episode_return += reward
                    ep_step += 1
                
                # after the rollout, preform optimization
                # use the critic to estimate the value of the next step
                with torch.no_grad():
                    _, next_value = self.model(torch_next_obs)
                
                # compute loss
                actor_loss, critic_loss, entropy_loss = self.A2Closs(l_params, self.buffer, next_value)
                loss = (l_params['critic_weight'] * critic_loss + l_params['actor_weight'] * actor_loss + l_params['entropy_weight'] * entropy_loss)
                
                # preform optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # log epoch stats
            self.writer.add_scalars(f'{self.experiment_phase}/Training Losses', {'Actor': actor_loss.item(),'Critic': critic_loss.item(),
                                    'Total': loss.item()}, self.episode)
            self.writer.add_scalar(f'{self.experiment_phase}/Training Episode Return', episode_return, self.episode)
            # self.writer.add_scalar('Returns/Rolling Average Return', rolling_avg_return, self.episode)
            # if sucessful, log steps to completion
            
            # preform evaluateion every 'eval_every_episodes'
            if self.episode % val_every_episodes == 0:
                self.model.eval()
                _, avg_reward, success_rate, steps_to_done = evaluate_agent_rewards(self.device, self.model, env, num_episodes=50)
                self.model.train()
                
                # log evaluation results
                self.writer.add_scalar(f'{self.experiment_phase}/Eval/Average Reward', avg_reward, self.episode)
                self.writer.add_scalar(f'{self.experiment_phase}/Eval/Success Rate', success_rate, self.episode)
                self.writer.add_scalar(f'{self.experiment_phase}/Eval/Steps to Done', steps_to_done, self.episode)
            
            # save every 1000 episodes
            if (self.episode + 1) % save_every_episodes == 0 and save_every_episodes is not None:
                checkpoint_filename = self.path + '\\' + f"{self.experiment_phase}_epi_{self.episode}.pth"
                torch.save(self.model.state_dict(), checkpoint_filename)
