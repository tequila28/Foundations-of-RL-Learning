"""Chapter 8: run the project's REINFORCE and Actor-Critic agents."""

import numpy as np
import torch

from Chapter8_Policy_Gradient.src.algorithms import REINFORCEAgent, ActorCriticAgent
from apps.gridworld_lab.common import evaluate_policy, rollout


def run(env, config):
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    algorithm = config["algorithm"]
    if algorithm == "REINFORCE":
        agent = REINFORCEAgent(env, hidden_size=config["hidden_size"], learning_rate=config["alpha"], gamma=config["gamma"], device="cpu")
    else:
        agent = ActorCriticAgent(env, hidden_size=config["hidden_size"], actor_lr=config["alpha"], critic_lr=config["alpha"], gamma=config["gamma"], device="cpu")
    stats = agent.train(n_episodes=config["episodes"], max_steps=config["max_steps"])
    policy = agent.get_policy()
    values = agent.get_value_function() if hasattr(agent, "get_value_function") else evaluate_policy(env, policy)
    rng = np.random.default_rng(seed + 1)
    path = rollout(env, policy, config["start"], config["max_steps"], rng)
    return {"policy": policy, "values": values, "rewards": stats["episode_rewards"], "path": path}
