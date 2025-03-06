import sys
import torch  
import torch.optim as optim
from tqdm import tqdm

# import gymnasium as gym
# import minigrid
# from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper, ImgObsWrapper, FullyObsWrapper, RGBImgPartialObsWrapper, ReseedWrapper, ActionBonus
import numpy as np  
# import matplotlib.pyplot as plt

class ReinforceTrainer(): 
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
        
        # built in optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
    
    def update_phase(self, phase):
        self.phase = phase
    
    def set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    # update policy
    def update_policy(self, rewards, log_probs, entropies):
        discounted_rewards = []
        
        # compute the rewards for each step
        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.gamma**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)
            
        discounted_rewards = torch.tensor(discounted_rewards)
        # normalize discounted rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
        
        # compute the E(prob * G)
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)
        policy_gradient = torch.stack(policy_gradient).sum()
        
        # compute 
        entropy_loss = self.entropy_weight * entropies.sum()
        loss = policy_gradient - entropy_loss
        
        # policy update step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # main loop
    def train(self, env, num_episodes, max_steps, entropy_weight= 1e-5):
        self.entropy_weight = entropy_weight
        
        # zero params
        all_numsteps = []
        all_rewards = []
        all_entropies = []
        
        for _ in range(num_episodes):
            self.episode += 1
            
            # reset state
            state, _ = env.reset()
            log_probs = []
            rewards = []
            entropies = []
            
            # rollout
            for steps in range(max_steps + 2): 
                # get policy output
                action, log_prob, entropy = self.policy_network.get_action(state)
                # preform step
                new_state, reward, terminated, truncated, _ = env.step(action)
                # store in buffer
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)
                # stop in either case
                terminated = terminated or truncated
                # if stopped, policy update
                if terminated:
                    entropies = torch.stack(entropies)
                    self.update_policy(rewards, log_probs, entropies)
                    # update data
                    all_numsteps.append(steps)
                    all_rewards.append(np.sum(rewards))
                    episode_entropy = entropies.mean().item()
                    all_entropies.append((episode_entropy))
                    
                    # log
                    self.logging(rewards, all_rewards, all_numsteps, all_entropies)
                    
                    # if terminal
                    break
                
                # update state if not terminal
                state = new_state
    
    def logging(self, rewards, all_rewards, all_numsteps, all_entropies):
        # log if TB exists
        if self.writer is not None:
            self.writer.add_scalars(f'{self.phase}/Training Rewards', {'Episode Reward': np.round(np.sum(rewards), decimals = 3),
                                                        'Avg Reward': np.round(np.mean(all_rewards[-100:]), decimals = 3)}, self.episode)
            self.writer.add_scalars(f'{self.phase}/Training Steps', {'Episode Steps': all_numsteps[-1],
                                                        'Avg Steps': np.round(np.mean(all_numsteps[-100:]), decimals=3)}, 
                                                        self.episode)
            self.writer.add_scalar(f'{self.phase}/Entropy', np.round(np.mean(all_entropies), decimals = 3), self.episode)
            self.writer.flush()
        else:
            printout_every = 10
            if self.episode % printout_every == 0:
                sys.stdout.write("episode: {}, last episode reward: {}, average_reward: {}, last episode length: {}, average length: {}, entropy: {}\n".format(self.episode,
                                np.round(np.sum(rewards), decimals = 3),  
                                np.round(np.mean(all_rewards[-100:]), decimals = 3), 
                                all_numsteps[-1], 
                                np.round(np.mean(all_numsteps[-100:]), decimals=3), 
                                np.round(np.stack(all_entropies[-100:]).mean().item(), decimals = 3)))
