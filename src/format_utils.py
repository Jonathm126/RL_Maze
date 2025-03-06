import torch

def preprocess_obs(obs, device):
    ''' preprocess an observation to torch image and cast to float. '''
    np_obs = torch.from_numpy(obs).type(torch.float32)
    np_obs = np_obs / 255.0 # scale to 0-1
    np_obs = np_obs.permute(2, 0, 1).unsqueeze(0).to(device)
    return np_obs

def map_action(action):
    ''' maps a 0-3 action space to 0-6, with 3, 4, 6 unused.'''
    # if tensor
    if isinstance(action, torch.Tensor):
        return torch.where(action == 3, torch.tensor(5), action)
    else:
        # if not a tensor
        return 5 if action == 3 else action