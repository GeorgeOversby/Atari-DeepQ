import glob
import time

import numpy as np

from src.parameters import AtariParameters
from src.setup import setup_for_evaluation

for model_name in glob.glob("models/*.pt"):
    model_name = model_name.replace("models\\", "").replace(".pt","")
    print("Running evaulation for %s" % model_name)

    params = AtariParameters.from_model_name(model_name)
    agent, runner = setup_for_evaluation(params,epsilon=0.05)
    agent.load_network(model_name)
    t1 = time.time()
    rewards = []
    for i in range(30):
        rewards += [runner.run_test()]
        print(i, rewards[-1])
    print("Results for %s" % model_name)
    print("Time to evaluate: %s" % (time.time() - t1))
    print('max:', np.max(rewards))
    print("mean:", np.mean(rewards))
    print("std:", np.std(rewards))
    print()
