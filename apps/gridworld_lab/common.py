"""Shared environment and policy helpers used by chapter experiment modules."""

import numpy as np

from Chapter1_Basic_Concepts.src.environment_model import GridWorld
from Chapter2_Bellman_Equations.src.algorithms.bellman_equation import BellmanSolver


def make_environment(config):
    size = config["size"]
    forbidden = [int(s) for s in config["forbidden"] if int(s) != int(config["target"]) and int(s) < size * size]
    return GridWorld(
        size=size,
        gamma=config["gamma"],
        forbidden_states=forbidden,
        target_states=[int(config["target"])],
        r_bound=config["r_wall"],
        r_forbid=config["r_forbid"],
        r_target=config["r_target"],
        r_default=config["r_step"],
    )


def soft_policy(q_values, epsilon):
    n_actions = q_values.shape[-1]
    policy = np.full_like(q_values, epsilon / n_actions, dtype=float)
    for state, row in enumerate(q_values):
        best = np.flatnonzero(np.isclose(row, np.max(row)))
        policy[state, best] += (1.0 - epsilon) / len(best)
    return policy


def rollout(env, policy, start, steps, rng, greedy=False):
    states = [int(start)]
    state = int(start)
    for _ in range(steps):
        if greedy:
            action = int(np.argmax(policy[state]))
        else:
            probabilities = policy[state] / policy[state].sum()
            action = int(rng.choice(env.n_actions, p=probabilities))
        state, _ = env.transition_logic(state, env.actions[action])
        states.append(int(state))
    return states


def evaluate_policy(env, policy, max_iterations=2000, tolerance=1e-7):
    return BellmanSolver(env).iterative_solution(policy, max_iterations, tolerance)
