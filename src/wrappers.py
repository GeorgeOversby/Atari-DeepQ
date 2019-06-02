from collections import deque
import gym
import numpy as np
from baselines.common.atari_wrappers import FireResetEnv, EpisodicLifeEnv
from gym import spaces
from PIL import Image
# Mostly copy-pasted from https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = 1
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs

class ProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None,top_cut=0):
        super(ProcessFrame, self).__init__(env)
        self.top_cut = top_cut
        self.observation_space = spaces.Box(low=0, high=255, shape=(102-top_cut, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame.process(obs,self.top_cut)

    @staticmethod
    def process(frame,top_cut):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        im = Image.fromarray(img)
        im = im.resize((84, 110),Image.BILINEAR)
        resized_screen = np.array(im)
        x_t = resized_screen[top_cut:102, :]
        x_t = np.reshape(x_t, [102-top_cut, 84, 1])
        return x_t.astype(np.uint8)

#baselines.common.atari_wrappers.LazyFrames
class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=0)
        if dtype is not None:
            out = out.astype(dtype)
        return out

#baselines.common.atari_wrappers.FrameStack
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=tuple([k] + list(shp)[:2]), dtype=np.float32)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ImageToPyTorch(gym.ObservationWrapper):

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)

class FireAfterLifeLost(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            obs, reward, done, info = self.env.step(1)
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        self.env.reset()
        ob,_,_,_ = self.env.step(1)
        return ob

def wrap_atari(env, stack_frames=4,training=True,crop_top=True):
    if training:
        env = EpisodicLifeEnv(env)

    env = NoopResetEnv(env)

    if 'FIRE' in env.unwrapped.get_action_meanings():
        if(training):
            env = FireResetEnv(env)
        else:
            env = FireAfterLifeLost(env)

    env = ProcessFrame(env,18 if crop_top else 0)
    env = ImageToPyTorch(env)
    env = FrameStack(env, stack_frames)
    return env