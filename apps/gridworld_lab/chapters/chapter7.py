"""Chapter 7: polynomial-feature linear TD(0) policy evaluation."""

import numpy as np


def run(env, config, rng):
    def features(state):
        x = (state % env.size) / max(env.size - 1, 1)
        y = (state // env.size) / max(env.size - 1, 1)
        return np.array([1.0, x, y, x * y, x * x, y * y])

    weights = np.zeros(6)
    policy = np.full((env.n_states, env.n_actions), 1 / env.n_actions)
    rewards, last_path = [], [config["start"]]
    for _ in range(config["episodes"]):
        state, total = int(config["start"]), 0.0
        path = [state]
        for _step in range(config["max_steps"]):
            action = int(rng.choice(env.n_actions))
            next_state, hit_wall = env.transition_logic(state, env.actions[action])
            reward = env.get_reward(next_state, hit_wall)
            current_features, next_features = features(state), features(next_state)
            td_error = reward + env.gamma * (next_features @ weights) - (current_features @ weights)
            weights += config["alpha"] * td_error * current_features
            state = next_state
            path.append(state)
            total += reward
        rewards.append(total)
        last_path = path
    values = np.array([features(state) @ weights for state in range(env.n_states)])
    return {"policy": policy, "values": values, "rewards": rewards, "path": last_path}
