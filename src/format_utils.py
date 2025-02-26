import torch

def preproess_obs(obs, device):
    ''' preprocess an observation to torch image and cast to float. '''
    return torch.tensor(obs, device=device, dtype=torch.float) / 255.0

def map_action(action):
    ''' maps a 0-3 action space to 0-6, with 3, 4, 6 unused.'''
    # if tensor
    if isinstance(action, torch.Tensor):
        return torch.where(action == 3, torch.tensor(5), action)
    else:
        # if not a tensor
        return 5 if action == 3 else action