from collections import namedtuple

import gym
from gym.envs.atari.atari_env import AtariEnv
from src import dqn_model
import src.wrappers as wrappers

MODELS_DIR = "models"

class Parameters():

    def __init__(self,env_name):
        self.env_name = env_name
        self.stop_reward = None
        self.epsilon_frames = None
        self.network_fn = None
        self.env_wrapper = None
        self.replay_initial = 10000
        self.replay_size = 1000000
        self.epsilon_start = 1.0
        self.epsilon_final = 0.02
        self.learning_rate = 0.00002
        self.gamma = 0.99
        self.batch_size = 32
        self.target_net_sync = 10000/4
        self.use_double_q = False
        self.use_prio = False

    @staticmethod
    def from_env_name(env_name):
        if(get_base_env_type(env_name) is AtariEnv):
            return AtariParameters(env_name)
        elif("CartPole" in env_name):
            return CartpoleParameters(env_name)
        else:
            return Parameters(env_name)


class AtariParameters(Parameters):

    def __init__(self, env_name):
        super().__init__(env_name)
        self.epsilon_frames = 10 ** 5
        self.env_wrapper = wrappers.wrap_atari
        self.network_fn = dqn_model.atari_net_fn()

        if("Pong" in env_name):
            self.replay_size = 100000
            self.learning_rate = 0.0001 * 3

        self.ArgsFromString = namedtuple('Args', 'big double dueling prio')

    def apply_args(self,args):
        self.use_double_q = args.double
        if (args.dueling):
            self.network_fn = dqn_model.atari_net_dueling_fn(args.big)
        else:
            self.network_fn = dqn_model.atari_net_fn(args.big)

    @staticmethod
    def from_model_name(model_name):
        game_name, args_string = model_name.split("_")
        params = AtariParameters(game_name)
        args = params.ArgsFromString(double="Double" in args_string, dueling="Dueling" in args_string,
                              big="Big" in args_string, prio="Prio" in args_string)
        params.apply_args(args)
        return params

class CartpoleParameters(Parameters):

    def __init__(self, env_name):
        super().__init__(env_name)
        self.epsilon_frames = 10 ** 2
        self.replay_initial = 200
        self.network_fn = dqn_model.feed_foward_fn(30)

        self.learning_rate = 0.01

def unpack_env(env_name):
    env = gym.make(env_name)
    result = [env]
    while(hasattr(env,"env")):
        result.append(env.env)
        env = env.env
    return result

def get_base_env_type(env_name):
    return type(unpack_env(env_name)[-1])
