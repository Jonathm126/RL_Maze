import torch.nn as nn
from torch.distributions import Categorical

class ActCrit(nn.Module):
    ''' an actor-critic model with two heads.'''
    def __init__(self, obs_space, action_space):
        super().__init__()
        
        # store params
        self.action_space = action_space
        self.final_conv_size = 9
        self.final_conv_depth = 64
        
        # build image processing backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1), # output 55 X 55
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # output 27 X 27
            
            nn.Conv2d(16, 32, kernel_size=4, stride=2), # 12 X 12
            nn.ReLU(),
            
            nn.Conv2d(32, self.final_conv_depth, kernel_size=4, stride=1), # 9 X 9
            nn.ReLU(),
            
            nn.Flatten(start_dim=1)
        )
        
        # actor network
        self.actor = nn.Sequential(
            nn.Linear((self.final_conv_size ** 2) * self.final_conv_depth, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space),
            nn.Softmax(dim=-1)
        )

        # critic_network
        self.critic = nn.Sequential(
            nn.Linear((self.final_conv_size ** 2) * self.final_conv_depth, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        # get action distribution
        act_proba = self.actor(features)
        act_dist = Categorical(probs = act_proba)
        # get value
        value = self.critic(features).squeeze(1)
        
        return act_dist, value