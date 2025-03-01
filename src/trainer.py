import torch
import torch.distributions 
import torch.nn.functional as F
from dataclasses import dataclass

# my imports
from src.format_utils import preprocess_obs, map_action
from src.eval_utils import evaluate_agent_rewards

@dataclass
class Trainer():
    def __init__(self, device, model, optimizer, writer, model_path = None):
        self.device = device
        self.episode = 0
        self.model = model
        self.optimizer = optimizer
        self.writer = writer
        self.path = model_path
    
    def reset_episodes(self):
        self.episode = 0
    
    def A2Closs(self, l_params, rollout, next_value):
        '''compute a2c losses for tensors of values'''
        # rollout size
        n_rollout = rollout['rewards'].shape[0]
        gamma = l_params['gamma']
        
        # compute returns
        returns = torch.cat([torch.zeros(n_rollout, device=self.device), 
                            next_value], dim=-1)
        for t in reversed(range(n_rollout)):
            returns[t] = rollout['rewards'][t] + gamma * returns[t + 1] * (1 - rollout['dones'][t]) 
        returns = returns[:-1]
        
        # compute the advantage function
        advantage = returns - rollout['values']
        # value loss (critic)
        critic_loss = F.mse_loss(rollout['values'], returns.detach(), reduction='mean')
        # actor loss 
        actor_loss = -(rollout['log_probs'] * advantage.detach()).mean()
        # entropy loss
        entropy_loss = rollout['entropies'].mean()
        
        return actor_loss, critic_loss, entropy_loss
    
    def A2CTrain(self, env, l_params, max_episodes, eval_every_episodes, rollout_size, experiment_phase = None, save_every = 1000):
        '''preformes rollout with rollout_size'''
        
        # compute episode
        start_episode = self.episode
        end_episode = max_episodes + start_episode
        
        # set optimizer and model
        self.model.train()
        
        # iterate on eposides
        for self.episode in range (start_episode, end_episode):
            # rollout buffer - dict of tensosrs
            rollout_buffer = {key: torch.zeros((rollout_size,), dtype=torch.float32, device=self.device) 
                                for key in ["log_probs", "values", "rewards", "dones", "entropies"]}            
            
            # get initial observation
            obs, _ = env.reset()
            torch_obs = preprocess_obs(obs, self.device)
            
            # reset params
            done, truncated = False, False
            episode_return = 0
            
            # preform steps
            while not done and not truncated:
                ep_step = 0
                # preform rollout
                for rollout_step in range(rollout_size):
                    # check if done
                    if done or truncated: 
                        break
                    
                    # get model outputs
                    act_dist, value = self.model(torch_obs)
                    
                    # compute action and params
                    action = act_dist.sample()
                    log_prob_action = act_dist.log_prob(action)
                    action_entropy = act_dist.entropy()
                    
                    # map action 3 to 5 (unused actions)
                    action_mapped = map_action(action).item()
                    self.writer.add_scalar(f'{experiment_phase}/Action Taken', action_mapped, self.episode)
                    
                    # preform step
                    next_obs, reward, done, truncated, _ = env.step(action_mapped)
                    torch_next_obs = preprocess_obs(next_obs, self.device)
                    reward = reward * l_params['reward_amplify'] - ep_step * l_params['reward_penalty']
                    
                    # append to rollout buffer
                    rollout_buffer["log_probs"][rollout_step] = log_prob_action
                    rollout_buffer["values"][rollout_step] = value
                    rollout_buffer["rewards"][rollout_step] = reward
                    rollout_buffer["dones"][rollout_step] = done
                    rollout_buffer["entropies"][rollout_step] = action_entropy
                    
                    torch_obs = torch_next_obs
                    episode_return += reward
                    ep_step += 1
                
                # after the rollout, preform optimization
                # use the critic to estimate the value of the next step
                with torch.no_grad():
                    _, next_value = self.model(torch_next_obs)
                
                # compute loss
                actor_loss, critic_loss, entropy_loss = self.A2Closs(l_params, rollout_buffer, next_value)
                loss = (l_params['critic_weight'] * critic_loss +
                        l_params['actor_weight'] * actor_loss +
                        l_params['entropy_weight'] * entropy_loss)
                
                # preform optimization step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # reset rollout
                rollout_buffer = {key: torch.zeros_like(tensor) for key, tensor in rollout_buffer.items()}
            
            print(self.episode)
            
            # log epoch stats
            self.writer.add_scalars(f'{experiment_phase}/Training Losses', {'Actor': actor_loss.item(),'Critic': critic_loss.item(),
                                    'Total': loss.item()}, self.episode)
            self.writer.add_scalar(f'{experiment_phase}/Training Episode Return', episode_return, self.episode)
            # self.writer.add_scalar('Returns/Rolling Average Return', rolling_avg_return, self.episode)
            
            # preform evaluateion every 'eval_every_episodes'
            if self.episode % eval_every_episodes == 0:
                self.model.eval()
                _, avg_reward, success_rate = evaluate_agent_rewards(self.device, self.model, env, num_episodes=50)
                self.model.train()
                
                # log evaluation results
                self.writer.add_scalar(f'{experiment_phase}/Eval/Average Reward', avg_reward, self.episode)
                self.writer.add_scalar(f'{experiment_phase}/Eval/Success Rate', success_rate, self.episode)
            
            # save every 1000 episodes
            if (self.episode + 1) % save_every == 0 and save_every is not None:
                checkpoint_filename = self.path + '\\' + f"{experiment_phase}_epi_{self.episode}.pth"
                torch.save(self.model.state_dict(), checkpoint_filename)
