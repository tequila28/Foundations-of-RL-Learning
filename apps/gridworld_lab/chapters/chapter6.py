"""Chapter 6: tabular SARSA, Expected SARSA, and Q-learning."""

import numpy as np

from apps.gridworld_lab.common import soft_policy


def run(env, config, rng):
    algorithm = config["algorithm"]
    q_values = np.zeros((env.n_states, env.n_actions), dtype=float)
    rewards, last_path = [], [config["start"]]
    final_epsilon = config["epsilon"]
    for episode in range(config["episodes"]):
        epsilon = max(config.get("epsilon_min", 0), config["epsilon"] * config.get("epsilon_decay", 1.0) ** episode)
        final_epsilon = epsilon
        state = int(config["start"])
        path, total = [state], 0.0
        action = int(rng.integers(env.n_actions))
        for step in range(config["max_steps"]):
            if rng.random() < epsilon:
                action = int(rng.integers(env.n_actions))
            elif algorithm != "SARSA" or step == 0:
                action = int(np.argmax(q_values[state]))
            next_state, hit_wall = env.transition_logic(state, env.actions[action])
            reward = env.get_reward(next_state, hit_wall)
            if algorithm == "SARSA":
                next_action = int(rng.integers(env.n_actions)) if rng.random() < epsilon else int(np.argmax(q_values[next_state]))
                target = reward + env.gamma * q_values[next_state, next_action]
            elif algorithm == "Expected SARSA":
                probabilities = soft_policy(q_values[next_state:next_state + 1], epsilon)[0]
                target = reward + env.gamma * float(probabilities @ q_values[next_state])
                next_action = int(np.argmax(q_values[next_state]))
            else:
                target = reward + env.gamma * float(np.max(q_values[next_state]))
                next_action = int(np.argmax(q_values[next_state]))
            q_values[state, action] += config["alpha"] * (target - q_values[state, action])
            state, action = next_state, next_action
            path.append(state)
            total += reward
        rewards.append(total)
        last_path = path
    return {"policy": soft_policy(q_values, final_epsilon), "values": np.max(q_values, axis=1), "value_label": "maxₐ Q(s,a)", "cell_value_label": "maxQ", "rewards": rewards, "path": last_path, "q_values": q_values}
