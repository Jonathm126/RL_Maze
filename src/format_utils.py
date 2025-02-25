import torch

def preproess_obs(obs, device):
    ''' preprocess an observation to torch image. '''
    return torch.tensor(obs, device=device, dtype=torch.float)

def map_action(action):
    ''' maps a 0-3 action space to 0-6, with 3, 4, 6 unused.'''
    # map 3->5, otherwise unchanged the mapping
    action_map = {0: 0, 1: 1, 2: 2, 3: 5}

    # Return the mapped action
    return action_map.get(action, None) 