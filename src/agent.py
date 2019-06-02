import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

from src.parameters import MODELS_DIR
from src.utils import to_numpy, to_tensor, cuda_if_possible, torch_load
import torch

class ActionModifier():

    def setup(self,action_size):
        self.action_size = action_size
        self.step(0)

    def __call__(self,actions):
        raise NotImplementedError

    def step(self,update):
        pass

class RandomActionModifier(ActionModifier):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self,actions):
        mask = np.random.random(size=len(actions)) < self.epsilon
        rand_actions = np.random.choice(self.action_size, sum(mask))
        actions[mask] = rand_actions
        return actions

class TrainingRandomActionModifier(RandomActionModifier):

    def __init__(self, epsilon_frames, epsilon_start, epsilon_final):
        super().__init__(epsilon_start)
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_frames = epsilon_frames

    def step(self, frame_idx):
        self.epsilon = max(self.epsilon_final, self.epsilon_start - frame_idx / self.epsilon_frames)

class Agent():

    def __init__(self,env_creator):
        self.obs_shape = env_creator.obs_shape
        self.action_size = env_creator.action_size

    def act(self, obs):
        raise NotImplementedError

    def get_networks(self):
        raise NotImplementedError

    def step(self, frame_idx):
        raise NotImplementedError

    def train_mode(self):
        for net in self.get_networks():
            net.train()
        self.training = True

    def eval_mode(self):
        for net in self.get_networks():
            net.eval()
        self.training = False

class DQNAgent(Agent):

    def __init__(self, env_creator, network_fn, action_modifier, gamma, lr, target_net_sync,double_q):
        super().__init__(env_creator)
        self.network = cuda_if_possible(network_fn(env_creator.obs_shape, env_creator.action_size))
        self.target_network = deepcopy(self.network)

        self.parallel_size = env_creator.parallel_size

        self.target_net_sync = target_net_sync
        self.gamma = gamma
        self.double_q = double_q

        action_modifier.setup(env_creator.action_size)
        self.training_action_modifier = action_modifier

        self.opt = optim.Adam(self.network.parameters(), lr=lr)
        self.name = "%s_%s" % (env_creator.name,self.network.name + "Double" if double_q else "")

    def act(self, obs):
        out = self.network(to_tensor(obs))
        out = to_numpy(out.argmax(-1))
        out = self.training_action_modifier(out)
        return out

    def train(self,minibatch):
        minibatch.rewards = np.clip(minibatch.rewards,-1,1)
        Q = self.network(to_tensor(minibatch.obs))
        targ_Q = minibatch.rewards + (self.gamma * self.calc_Q_next_max(minibatch) * ~minibatch.dones)

        losses = to_tensor(minibatch.weights).float() * F.mse_loss(Q[range(len(Q)), minibatch.actions] , to_tensor(targ_Q).float().detach(),reduce=False)
        loss = losses.mean()

        self.minimize(loss)
        return to_numpy(losses)

    def calc_Q_next_max(self,minibatch):
        if(self.double_q):
            Q_next_argmax = self.network(to_tensor(minibatch.next_obs)).argmax(1)
            Q_next_max = self.target_network(to_tensor(minibatch.next_obs))
            return to_numpy(Q_next_max[range(len(Q_next_max)), Q_next_argmax])
        else:
            return to_numpy(self.target_network(to_tensor(minibatch.next_obs))).max(1)

    def minimize(self,loss):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def get_networks(self):
        return [self.target_network,self.network]

    def sync_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def step(self, frame_idx):
        if frame_idx % self.target_net_sync < self.parallel_size:
            self.sync_target_network()
        self.training_action_modifier.step(frame_idx)

    def save_network(self,comment=""):
        torch.save(self.network.state_dict(), MODELS_DIR + '/%s.pt' % (self.name+comment))

    def load_network(self,name=""):
        self.network.load_state_dict(torch_load(MODELS_DIR + '/%s.pt' % (name)))
