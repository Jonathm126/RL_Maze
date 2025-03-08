import sys
import torch  
from torch.optim import Adam
import numpy as np  

from src.train_utils import Memory

class A2CTrainer(): 
    def __init__(self, device, policy_network, lr = 3e-4, gamma = 0.99, writer = None):
        # store params
        self.device = device
        self.writer = writer
        self.policy_network = policy_network
        self.episode = 0
        self.lr = lr
        self.gamma = gamma
        self.phase = None
        self.path = None
        self.memory = Memory()
        
        # built in optimizer
        self.optimizer = Adam(params = self.policy_network.parameters(), lr=lr)
    
    def update_phase(self, phase):
        self.phase = phase
    
    def set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
    def zero_episodes(self):
        self.episode = 0
    
    # update policy
    def update_policy(self, q_val):
        # to torch
        values = torch.stack(self.memory.values).squeeze()
        q_vals = np.zeros((len(self.memory), 1))
        log_probs = torch.stack(self.memory.log_probs)
        entropies = torch.stack(self.memory.entropies)
        
        # compute the return (q)
        for i, (_, _, reward, done, _) in enumerate(self.memory.reversed()):
            q_val = reward + self.gamma* q_val*(1.0-done)
            q_vals[len(self.memory)-1 - i] = q_val 
        
        # compute advantage
        # discounted_returns = compute_returns(self.gamma, rewards)
        advantages = torch.from_numpy(q_vals).to(self.device) - values
        
        # actor loss
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # critic loss
        critic_loss = advantages.pow(2).mean()
        
        # entropy loss 
        entropy_loss = self.entropy_weight * entropies.mean()
        
        # policy update step
        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss - entropy_loss
        loss.backward()
        self.optimizer.step()

    # main loop
    def train(self, env, num_episodes, entropy_weight= 1e-5, n_rollout = 5):
        self.entropy_weight = entropy_weight
        self.n_rollout = n_rollout
        
        # reset
        all_rewards = []
        all_entropies = []
        all_steps = []
        
        for _ in range(num_episodes):
            self.episode += 1
            
            # reset state
            state, _ = env.reset()
            
            # empty buffer
            done, truncated = False, False
            episode_reward, steps, episode_entropy = 0, 0, 0
            
            # preform rollout
            while not done:
                # get policy output
                action, log_prob, entropy, value = self.policy_network.get_action_and_value(state)
                
                # preform step
                new_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                
                # store in buffer
                episode_reward += reward
                episode_entropy += entropy
                steps +=1   
                self.memory.add(log_prob, value, reward, done, entropy)                
                
                # update state
                state = new_state
                
                # if stopped or rollout done, policy update
                if done or (steps % self.n_rollout == 0):
                    _, _, _, value = self.policy_network.get_action_and_value(state)
                    last_q_val = value.detach().data.cpu().numpy()
                    self.update_policy(last_q_val)
                    self.memory.clear()
            
            # update data
            all_steps.append(steps)
            all_rewards.append(reward)
            all_entropies.append(episode_entropy.item() / steps)
            
            # log
            self.logging(all_steps, all_rewards, all_entropies)
    
    
    def logging(self, all_steps, all_rewards, all_entropies):
        avg_100_steps = np.round(np.mean(all_steps[-100:]), decimals=3)
        avg_100_rewards = np.round(np.mean(all_rewards[-100:]), decimals=3)        
        
        # log if TB exists
        if self.writer is not None:
            self.writer.add_scalars(f'{self.phase}/Training Rewards', {'Episode Reward': all_rewards[-1],
                                                                        'Avg Reward': avg_100_rewards},
                                                                        self.episode)
            self.writer.add_scalars(f'{self.phase}/Training Steps', {'Episode Steps': all_steps[-1],
                                                                        'Avg Steps': avg_100_steps}, 
                                                                        self.episode)
            self.writer.add_scalar(f'{self.phase}/Entropy', all_entropies[-1], self.episode)
            self.writer.flush()
        else:
            printout_every = 10
            if self.episode % printout_every == 0:
                sys.stdout.write("episode: {}, last episode reward: {}, average_reward: {}, last episode length: {}, average length: {}, entropy: {}\n".format(self.episode,
                                all_rewards[-1],  
                                avg_100_rewards, 
                                all_steps[-1], 
                                avg_100_steps, 
                                all_entropies[-1]))
