"""Chapter 1: inspect the shared GridWorld under a random policy."""

import numpy as np

from apps.gridworld_lab.common import evaluate_policy, rollout


def run(env, config, rng):
    policy = np.full((env.n_states, env.n_actions), 1 / env.n_actions)
    values = evaluate_policy(env, policy)
    path = rollout(env, policy, config["start"], config["max_steps"], rng)
    return {"policy": policy, "values": values, "rewards": [], "path": path}
