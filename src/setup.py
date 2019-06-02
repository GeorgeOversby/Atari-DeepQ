from collections import namedtuple

from src.agent import RandomActionModifier, DQNAgent, TrainingRandomActionModifier
from src.env_creator import EnvCreator
from src.parameters import AtariParameters
from src.runner import Runner

def setup(params,epsilon_action_modifier,parallel_size):
    env_creator = EnvCreator(params.env_name, parallel_size, wrapper=params.env_wrapper, seed=12)

    agent = DQNAgent(env_creator, params.network_fn, epsilon_action_modifier,
                     params.gamma, params.learning_rate, params.target_net_sync, params.use_double_q)
    runner = Runner(env_creator, agent)
    return agent,runner

def setup_for_evaluation(params,epsilon):
    epsilon_action_modifier = RandomActionModifier(epsilon)
    return setup(params,epsilon_action_modifier,parallel_size=1)

def setup_for_train(params):
    epsilon_action_modifier = TrainingRandomActionModifier(params.epsilon_frames, params.epsilon_start, params.epsilon_final)
    return setup(params,epsilon_action_modifier,parallel_size=4)
