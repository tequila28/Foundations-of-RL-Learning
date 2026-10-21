"""Chapter 2: evaluate a policy with Bellman iteration or a closed form."""

import numpy as np

from Chapter2_Bellman_Equations.src.algorithms.bellman_equation import BellmanSolver
from apps.gridworld_lab.common import evaluate_policy, rollout


def run(env, config, rng):
    policy = np.full((env.n_states, env.n_actions), 1 / env.n_actions)
    if config["algorithm"].endswith("闭式解"):
        values = BellmanSolver(env).closed_form_solution(policy)
    else:
        values = evaluate_policy(env, policy)
    path = rollout(env, policy, config["start"], config["max_steps"], rng)
    return {"policy": policy, "values": values, "rewards": [], "path": path}
