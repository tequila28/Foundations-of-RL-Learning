"""Chapter 5: run the repository's original SGD, MBGD, and BGD estimators."""

import numpy as np

from Chapter5_Stochastic_Approximation.src.environment import Environment
from Chapter5_Stochastic_Approximation.src.algorithms.sgd import SGD
from Chapter5_Stochastic_Approximation.src.algorithms.mbgd import MBGD
from Chapter5_Stochastic_Approximation.src.algorithms.bgd import BGD


def run(config):
    rng = np.random.default_rng(config["seed"])
    np.random.seed(config["seed"])
    square_size = config.get("square_size", 30)
    env = Environment(square_size=square_size, num_samples=config["samples"], total_iterations=config["episodes"])
    if config.get("use_original_sampling", False):
        env.generate_samples()
        target_label = "True mean"
    else:
        env.samples = rng.uniform(-square_size / 2, square_size / 2, size=(config["samples"], 2))
        env.true_mean = env.samples.mean(axis=0)
        target_label = "Sample mean"
    initial = np.array([config.get("init_x", 12.0), config.get("init_y", -12.0)])
    algorithm = config["algorithm"]
    if algorithm == "SGD":
        trajectory, errors = SGD(env).estimate(initial, alpha_type="constant", constant_alpha=config["alpha"])
    elif algorithm == "Mini-batch GD":
        trajectory, errors = MBGD(env).estimate(initial, batch_size=config["batch_size"], alpha_type="constant", constant_alpha=config["alpha"])
    else:
        trajectory, errors = BGD(env).estimate(initial, alpha_type="constant", constant_alpha=config["alpha"])
    return {"kind": "estimation", "samples": env.samples, "target": env.true_mean, "target_label": target_label,
            "trajectory": trajectory, "errors": errors}
