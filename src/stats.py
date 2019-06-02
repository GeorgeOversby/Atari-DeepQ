import datetime
import sys
from collections import deque
import time
import numpy as np

class StatsTracker():

    def __init__(self,parallel_size,agent,env_name):
        self.agent = agent
        self.cum_rewards = np.zeros(parallel_size)
        self.finished_rewards = deque(maxlen=100)

        self.frame_count = 0
        self.last_frame_count = 0
        self.games_played = 0
        self.mean_reward = None

        self.initial_time = time.time()
        self.last_time = time.time()

        self.callbacks = [callback(self,env_name) for callback in [ConsoleLogger,ModelSaver]]

    def track(self, obs, action, reward, next_obs, done):
        self.running_time = str(datetime.timedelta(seconds=time.time() - self.initial_time))
        self.cum_rewards += reward
        self.track_finished_games(done)

        self.frame_count += len(obs)
        if(self.frame_count % 500 == 0):
            if(self.finished_rewards):
                self.mean_reward = np.mean(self.finished_rewards)
            self.calculate_fps()

        for callback in self.callbacks:
            callback.on_frame_step()

    def track_finished_games(self, done):
        finished_games = list(self.cum_rewards[done])
        self.games_played += len(finished_games)
        self.finished_rewards += finished_games
        self.cum_rewards[done] = 0

    def calculate_fps(self):
        self.fps = (self.frame_count - self.last_frame_count) / (time.time() - self.last_time)
        self.last_time = time.time()
        self.last_frame_count = self.frame_count

    def end_training(self):
        for callback in self.callbacks:
            callback.on_training_end()


class Callback():

    def __init__(self,stats,run_name):
        self.stats = stats
        self.name = run_name
        self.agent = self.stats.agent

    def on_frame_step(self):
        pass

    def on_training_end(self):
        pass

class Logger(Callback):

    def on_frame_step(self):
        if(self.stats.frame_count % 500 == 0):
            self.log()

    def log(self):
        raise NotImplementedError

class ConsoleLogger(Logger):

    def __init__(self, stats, run_name):
        super().__init__(stats, run_name)
        self.items_to_output = []

    def log(self):
        self.add_float("average reward",self.stats.mean_reward)
        self.add_float("fps", self.stats.fps)
        self.add_float("epsilon",self.agent.training_action_modifier.epsilon)
        self.add_string("time", self.stats.running_time)

        self.flush_output()

    def on_training_end(self):
        print("Done!")

    def add_string(self, name, value):
        self.items_to_output.append("%s %s" % (name,value))

    def add_float(self, name, value, precision=2):
        f_string = f"%s %.{precision}f"
        if(value):
            self.items_to_output.append(f_string % (name, value))
        else:
            self.add_string(name,"None")

    def flush_output(self):
        details = ", ".join(self.items_to_output)
        print("%s: %s" % (self.stats.frame_count, details))
        self.items_to_output = []
        sys.stdout.flush()

class ModelSaver(Callback):

    def on_frame_step(self):
        if(self.stats.frame_count % 500_000 == 0):
            self.agent.save_network()
            print("saved")
            
    def on_training_end(self):
        self.agent.save_network()
