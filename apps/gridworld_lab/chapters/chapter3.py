"""Chapter 3: dynamic-programming planning algorithms."""

import numpy as np

from apps.gridworld_lab.common import rollout


def solve(env, algorithm, theta, max_iterations, inner_iterations):
    values = np.zeros(env.n_states)
    policy_ids = np.full(env.n_states, env.actions.index("stay"), dtype=int)

    def q_values(state):
        return env.R[state] + env.gamma * env.P[state] @ values

    if algorithm == "价值迭代":
        for _ in range(max_iterations):
            old_values = values.copy()
            for state in range(env.n_states):
                values[state] = np.max(env.R[state] + env.gamma * env.P[state] @ old_values)
            if np.max(np.abs(values - old_values)) < theta:
                break
        policy_ids = np.array([int(np.argmax(q_values(s))) for s in range(env.n_states)])
    else:
        for _ in range(max_iterations):
            stable = True
            evaluation_steps = inner_iterations if algorithm == "截断策略迭代" else 1000
            for _ in range(evaluation_steps):
                old_values = values.copy()
                for state in range(env.n_states):
                    action = policy_ids[state]
                    values[state] = env.R[state, action] + env.gamma * env.P[state, action] @ values
                if np.max(np.abs(values - old_values)) < theta:
                    break
            for state in range(env.n_states):
                new_action = int(np.argmax(q_values(state)))
                stable &= new_action == policy_ids[state]
                policy_ids[state] = new_action
            if stable:
                break
    return policy_ids, values


def run(env, config, rng):
    policy_ids, values = solve(env, config["algorithm"], config["theta"], config["iterations"], config["inner_iterations"])
    policy = np.eye(env.n_actions)[policy_ids]
    path = rollout(env, policy, config["start"], config["max_steps"], rng, greedy=True)
    return {"policy": policy, "values": values, "rewards": [], "path": path}
