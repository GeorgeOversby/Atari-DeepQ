import numpy as np
import torch
import torch.backends.cudnn
import gym

def set_seed(SEED,env=None):
    torch.manual_seed(SEED); np.random.seed(SEED);gym.spaces.prng.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    if(env):
        env.seed(SEED)

def cuda_if_possible(x):
    if(torch.cuda.is_available()):
        return x.cuda()
    return x

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def to_tensor(numpy):
    return cuda_if_possible(torch.from_numpy(numpy))

def torch_load(f):
    if(torch.cuda.is_available()):
        return torch.load(f)
    return torch.load(f,map_location='cpu')