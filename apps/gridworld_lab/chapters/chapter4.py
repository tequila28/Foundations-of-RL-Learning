"""Chapter 4: Monte Carlo control with epsilon-greedy or exploring starts."""

import numpy as np

from apps.gridworld_lab.common import soft_policy


def run(env, config, rng):
    algorithm = config["algorithm"]
    q_values = np.zeros((env.n_states, env.n_actions), dtype=float)
    counts = np.zeros_like(q_values)
    policy = np.full_like(q_values, 1 / env.n_actions)
    rewards, last_path = [], [config["start"]]
    for _ in range(config["episodes"]):
        state = int(rng.integers(env.n_states)) if algorithm == "MC Exploring Starts" else int(config["start"])
        action = int(rng.integers(env.n_actions)) if algorithm == "MC Exploring Starts" else int(rng.choice(env.n_actions, p=policy[state]))
        episode, path, total = [], [state], 0.0
        for _step in range(config["max_steps"]):
            next_state, hit_wall = env.transition_logic(state, env.actions[action])
            reward = env.get_reward(next_state, hit_wall)
            episode.append((state, action, reward))
            total += reward
            state = next_state
            path.append(state)
            action = int(rng.integers(env.n_actions)) if algorithm == "MC Exploring Starts" else int(rng.choice(env.n_actions, p=policy[state]))
        value_return, visited = 0.0, set()
        for state, action, reward in reversed(episode):
            value_return = reward + env.gamma * value_return
            pair = (state, action)
            if pair not in visited:
                counts[pair] += 1
                q_values[pair] += (value_return - q_values[pair]) / counts[pair]
                visited.add(pair)
        policy = soft_policy(q_values, config["epsilon"])
        rewards.append(total)
        last_path = path
    return {"policy": policy, "values": np.max(q_values, axis=1), "value_label": "maxₐ Q(s,a)", "cell_value_label": "maxQ", "rewards": rewards, "path": last_path, "q_values": q_values}
