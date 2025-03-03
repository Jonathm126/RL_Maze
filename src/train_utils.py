import torch

class RolloutBuffer():
    ''' helper class for the rollout buffer. '''
    def __init__(self, device, rollout_size):
        self.buffer_keys = ["log_probs", "values", "rewards", "dones", "entropies"]
        # init the buffer
        self.device = device
        self.buffer = {key: torch.zeros((rollout_size,), dtype=torch.float32, device=self.device) for key in self.buffer_keys}  
        self.position = 0
        self.capacity = rollout_size
    
    def reset(self):
        self.buffer = {key: torch.zeros((self.capacity,), dtype=torch.float32, device=self.device) for key in self.buffer_keys}
        self.position = 0
        return False
        
    def add(self, log_prob, value, reward, done, entropy):
        '''Add a new step's data to the buffer.'''
        if self.position >= self.capacity:
            raise IndexError("RolloutBuffer is full. Consider resetting or increasing the capacity.")
        
        self.buffer["log_probs"][self.position] = log_prob
        self.buffer["values"][self.position]    = value
        self.buffer["rewards"][self.position]   = reward
        self.buffer["dones"][self.position]     = done
        self.buffer["entropies"][self.position] = entropy
        
        self.position += 1
        return self.position == self.capacity
    
    def __getitem__(self, key):
        '''Retrieve the filled portion of the specified parameter.'''
        if key not in self.buffer_keys:
            raise KeyError(f"Parameter '{key}' not found in the buffer.")
        # Return only the valid (filled) portion of the tensor.
        return self.buffer[key][:self.position]