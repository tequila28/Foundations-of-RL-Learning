"""Chapter 5: run the repository's original SGD, MBGD, and BGD estimators."""

import numpy as np

from Chapter5_Stochastic_Approximation.src.environment import Environment
from Chapter5_Stochastic_Approximation.src.algorithms.sgd import SGD
from Chapter5_Stochastic_Approximation.src.algorithms.mbgd import MBGD
from Chapter5_Stochastic_Approximation.src.algorithms.bgd import BGD


def run(config):
    rng = np.random.default_rng(config["seed"])
    np.random.seed(config["seed"])
    env = Environment(square_size=30, num_samples=config["samples"], total_iterations=config["episodes"])
    env.samples = rng.uniform(-15, 15, size=(config["samples"], 2))
    env.true_mean = env.samples.mean(axis=0)
    initial = np.array([12.0, -12.0])
    algorithm = config["algorithm"]
    if algorithm == "SGD":
        trajectory, errors = SGD(env).estimate(initial, alpha_type="constant", constant_alpha=config["alpha"])
    elif algorithm == "Mini-batch GD":
        trajectory, errors = MBGD(env).estimate(initial, batch_size=config["batch_size"], alpha_type="constant", constant_alpha=config["alpha"])
    else:
        trajectory, errors = BGD(env).estimate(initial, alpha_type="constant", constant_alpha=config["alpha"])
    return {"kind": "estimation", "samples": env.samples, "target": env.true_mean, "trajectory": trajectory, "errors": errors}
