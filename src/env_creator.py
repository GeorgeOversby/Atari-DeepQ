from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym
import torch
import numpy as np
from src.utils import set_seed
import time
class EnvCreator():

    def __init__(self,name,parallel_size,wrapper=None,seed=None):
        self.name = name
        self.parallel_size = parallel_size
        self.wrapper = wrapper
        self.seed = seed

        self.create_training_env()

    def create_training_env(self):
        if(self.parallel_size == 1):
            self.env = SingleEnv(self.get_new_env())
        else:
            self.env = SubprocVecEnv([self.get_new_env for _ in range(self.parallel_size)])

        if(self.seed):
            set_seed(self.seed)

    def get_new_env(self,training=True):
        env = gym.make(self.name)
        if(self.wrapper):
            env = self.wrapper(env,training=training)
        return env

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def obs_shape(self):
        return self.env.observation_space.shape

class SingleEnv(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self,actions):
        next_obs ,reward, done , info = self.env.step(actions[0])
        if(done):
            next_obs = self.env.reset()
        return np.expand_dims(next_obs, 0),np.array([reward]),np.array([done]) ,info

    def reset(self):
        return np.expand_dims(self.env.reset(), 0)
