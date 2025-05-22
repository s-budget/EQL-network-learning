import json
import os

import numpy as np

from Seminar import symbolic_functions
from Seminar.LearningInstance import LearningInstance

if __name__ == "__main__":


    kwargs = {
        "results_dir": "results/benchmark/test",
        "n_layers": 2,
        "reg_weight": 5e-3,
        "learning_rate": 1e-3,
        "n_epochs1": 10001,
        "n_epochs2": 10001,
        "freezing_epoch_distance":500}

    if not os.path.exists(kwargs['results_dir']):
        os.makedirs(kwargs['results_dir'])
    with open(os.path.join(kwargs['results_dir'], 'args.txt'), 'w') as meta:
        meta.write(json.dumps(kwargs))

    instance = LearningInstance(**kwargs)
    instance.run_experiment(lambda x,y: x*np.sin(2*np.pi*y), "6x+2x2+5x3", 10)
