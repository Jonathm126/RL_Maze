import torch.nn as nn

class ActCrit(nn.Module):
    ''' an actor-critic model with two heads.'''
    def __init__(self, obs_space, action_space):
        super().__init__()
        
        # store params
        self.h, self.w, self.c = obs_space.shape[0:3]
        self.action_space = action_space
        self.final_conv_size = 25
        self.final_conv_depth = 64
        
        # build image processing backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, self.final_conv_depth, (2, 2)),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # actor network
        self.actor = nn.Sequential(
            nn.Linear(self.final_conv_size ** 2 * self.final_conv_depth, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_space),
            nn.Softmax(dim=-1)
        )

        # critic_network
        self.critic = nn.Sequential(
            nn.Linear(self.final_conv_size ** 2 * self.final_conv_depth, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        x = self.backbone(x)
        features = x.reshape(x.shape[0], -1)
        # get actor-critic heads
        act_dist = self.actor(features)
        value = self.critic(features).squeeze(1)
        
        return act_dist, value

class ACAgent():
    pass