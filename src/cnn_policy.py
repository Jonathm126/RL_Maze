import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# my imports
from src.format_utils import preprocess_obs

class CNNPolicyNetwork(nn.Module):
    def __init__(self, device, num_actions):
        super(CNNPolicyNetwork, self).__init__()
        
        # store params
        self.device = device
        self.final_conv_depth = 64
        self.hidden_size = 128
        self.num_actions = num_actions
        
        # feature extractor
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1), # output 55 X 55
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # output 27 X 27
            
            nn.Conv2d(16, 32, kernel_size=4, stride=2), # 12 X 12
            nn.ReLU(),
            
            nn.Conv2d(32, self.final_conv_depth, kernel_size=4, stride=1), # 9 X 9
            nn.ReLU(),
            
            nn.Flatten(start_dim=1)
        ).to(self.device)

        # build linear layers
        self.linear1 = nn.Linear((9 ** 2) * self.final_conv_depth, self.hidden_size).to(self.device)
        self.linear2 = nn.Linear(self.hidden_size, num_actions).to(self.device)        

    def forward(self, state):
        features = self.backbone(state)
        x = F.relu(self.linear1(features))
        x = F.softmax(self.linear2(x), dim=1)
        return x 
    
    def get_action(self, state):
        # state = torch.from_numpy(state).float().unsqueeze(0)
        # transform the state to tensor
        torch_state = preprocess_obs(state, self.device)
        # compute probabilities
        probs = self.forward(Variable(torch_state))
        # choose action
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().cpu().numpy()))
        # compute log prob and entropy
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return highest_prob_action, log_prob, entropy