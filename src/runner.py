from collections import namedtuple
import numpy as np
import time
from gym.wrappers import Monitor
from src.stats import StatsTracker

SingleExperience = namedtuple('SingleExperience', 'obs actions rewards next_obs dones')

class Queue():

    def __init__(self, maxlen):
        self.max_size = maxlen
        self.buffer = []
        self.pos = 0


    def append(self, item):
        if len(self.buffer) < self.max_size:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.max_size


    def __iadd__(self, data):
        for item in data:
            self.append(item)
        return self

    def __getitem__(self, item):
        return self.buffer[item]

    def __len__(self):
        return len(self.buffer)

class PrioritizedQueue(Queue):

    def __init__(self, maxlen):
        super().__init__(maxlen)
        self.priorities = np.zeros((maxlen,), dtype=np.float32)

    def append(self, item):
        max_prio = self.priorities.max() if self.buffer else 1.0
        self.priorities[self.pos] = max_prio
        return super().append(item)


class Experience():
    def __init__(self, max_size):
        self.max_size = max_size

        self.obs = Queue(maxlen=max_size)
        self.next_obs = Queue(maxlen=max_size)
        self.actions = Queue(maxlen=max_size)
        self.rewards = Queue(maxlen=max_size)
        self.dones = Queue(maxlen=max_size)

    def __iadd__(self, other):
        self.obs += list(other.obs)
        self.next_obs += list(other.next_obs)
        self.actions += list(other.actions)
        self.rewards += list(other.rewards)
        self.dones += list(other.dones)
        return self

    def __len__(self):
        return len(self.obs)

    def get_minibatch(self, batch_size):
        return Minibatch(self, batch_size)

    def sample_indexes(self, size):
        return np.random.choice(len(self.obs), size, replace=True)


class PrioritizedExperience(Experience):

    def __init__(self, max_size):
        super().__init__(max_size)
        self.obs = PrioritizedQueue(maxlen=max_size)
        self.beta = 0.4

    @property
    def priorities(self):
        return self.obs.priorities

    def sample_indexes(self, size):
        if len(self.obs.buffer) == self.max_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.obs.pos]
        probs = prios ** 0.6
        probs /= probs.sum()
        indices = np.random.choice(len(self.obs.buffer), size, p=probs)
        weights = (len(self.obs.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return indices,weights

    def update_priorities(self, idxs, values):
        self.priorities[idxs] = values
        # print(idxs,values)

class Minibatch():

    def __init__(self,experience,size):
        if(isinstance(experience,PrioritizedExperience)): # todo clean
            self.data_idxs, self.weights = experience.sample_indexes(size)
        else:
            self.data_idxs = experience.sample_indexes(size)
            self.weights = np.ones(size)

        self.obs      =  self.process(experience.obs)
        self.next_obs =  self.process(experience.next_obs)
        self.actions  =  self.process(experience.actions)
        self.rewards  =  self.process(experience.rewards)
        self.dones    =  self.process(experience.dones)

    def process(self,data):
        return np.array([data[idx] for idx in self.data_idxs])


class Runner():
    def __init__(self, env_creator, agent):
        self.env_creator = env_creator
        self.agent = agent
        self.env = env_creator.env
        self.obs = self.env.reset()
        self.tracker = StatsTracker(self.env_creator.parallel_size,agent,self.env_creator.name)

    def run(self,nsteps):
        self.agent.train_mode()
        experience = Experience(nsteps*self.env_creator.parallel_size)
        for j in range(nsteps):
            action = self.agent.act(self.obs)
            next_obs, reward, done, _ = self.env.step(action)
            experience += SingleExperience(self.obs, action, reward, next_obs, done)

            self.tracker.track(self.obs, action, reward, next_obs, done)
            self.obs = next_obs

        return experience

    def run_test(self,video_folder_name = None,render=False):
        self.agent.eval_mode()
        env = self.get_test_env(video_folder_name)

        obs = env.reset()
        total_reward  = 0
        done = False
        while not done:
            action = self.agent.act(np.expand_dims(obs,0))
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if(render):
                env.render()
                time.sleep(0.01)
        return total_reward

    def get_test_env(self,video_folder_name):
        env = self.env_creator.get_new_env(training=False)
        if(video_folder_name):
            env.metadata['video.frames_per_second'] = 30
            env =  Monitor(env, './videos/' + video_folder_name)
        return env

    def mean_reward_less_than(self,x):
        return not self.tracker.mean_reward or self.tracker.mean_reward < x

    def step(self):
        self.agent.step(self.tracker.frame_count)

    def end_training(self):
        self.tracker.end_training()