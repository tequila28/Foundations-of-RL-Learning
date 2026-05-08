# main.py
import numpy as np
import os
import sys
import argparse

import torch

# Environment path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(os.path.dirname(parent_dir))

# Ensure the results directory exists
results_dir = os.path.join(parent_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

from Chapter1_Basic_Concepts.src.environment_model import GridWorld
from Chapter8_Policy_Gradient.src.algorithms import REINFORCEAgent
from Chapter8_Policy_Gradient.src.algorithms import ActorCriticAgent
from Chapter8_Policy_Gradient.src.visualization import GridWorldVisualizer


def parse_arguments():
    """
    Parse command-line arguments for the GridWorld RL experiments

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="GridWorld Policy Gradient Experiments")

    # Environment parameters
    parser.add_argument("--size", type=int, default=5,
                       help="Grid size (default: 5)")
    parser.add_argument("--forbidden_states", nargs='+', type=int,
                       default=[6, 7, 12, 16, 18, 21],
                       help="Forbidden states (default: 6 7 12 16 18 21)")
    parser.add_argument("--target_states", nargs='+', type=int,
                       default=[17],
                       help="Target states (default: 17)")

    # Reward settings
    parser.add_argument("--r_bound", type=float, default=-1,
                       help="Boundary reward (default: -1)")
    parser.add_argument("--r_forbid", type=float, default=-1,
                       help="Forbidden state reward (default: -1)")
    parser.add_argument("--r_target", type=float, default=1,
                       help="Target state reward (default: 1)")
    parser.add_argument("--r_default", type=float, default=0,
                       help="Default reward (default: 0)")
    parser.add_argument("--gamma", type=float, default=0.9,
                       help="Discount factor")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    # Algorithm parameters
    parser.add_argument("--n_episodes", type=int, default=5000,
                       help="Number of training episodes (default: 1000)")
    parser.add_argument("--max_steps", type=int, default=20,
                       help="Maximum steps per episode (default: 500)")
    parser.add_argument("--hidden_size", type=int, default=128,
                       help="Hidden layer size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    
    # Feature extraction parameters
    parser.add_argument("--feature_type", type=str, default="one_hot",
                       choices=["one_hot", "grid_position", "coordinate", "random_projection"],
                       help="Type of feature extraction for Actor-Critic (default: one_hot)")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                       help="GAE lambda parameter for Actor-Critic (default: 0.95)")

    return parser.parse_args()


def create_environment(args):
    """
    Create GridWorld environment based on command-line arguments

    Args:
        args: Parsed command-line arguments

    Returns:
        GridWorld: Configured environment
    """
    env = GridWorld(
        size=args.size,
        gamma=args.gamma,
        forbidden_states=args.forbidden_states,
        target_states=args.target_states,
        r_bound=args.r_bound,
        r_forbid=args.r_forbid,
        r_target=args.r_target,
        r_default=args.r_default
    )
    return env


def train_reinforce(env, args, visualizer):
    """
    Train REINFORCE algorithm

    Args:
        env: GridWorld environment
        args: Command-line arguments
        visualizer: Visualization object

    Returns:
        dict: Training statistics
    """
    print("\n" + "=" * 60)
    print("Training REINFORCE Algorithm")
    print("=" * 60)

    # Initialize REINFORCE agent
    reinforce_agent = REINFORCEAgent(
        env=env,
        hidden_size=args.hidden_size,
        learning_rate=args.lr,
        gamma=args.gamma,
    )

    # Train the agent
    stats = reinforce_agent.train(
        n_episodes=args.n_episodes,
        max_steps=args.max_steps
    )

    # Get final policy
    policy = reinforce_agent.get_policy()

    visualizer.plot_training_curves(
        episode_rewards=stats['episode_rewards'],
        episode_lengths=stats['episode_lengths'],
        policy_entropies=stats.get('policy_entropies', []),  # Use get to prevent KeyError
        title="REINFORCE Training Curves",
        save=True
    )

    return {
        'policy': policy,
        'stats': stats
    }


def train_actor_critic(env, args, visualizer):
    """
    Train Actor-Critic algorithm

    Args:
        env: GridWorld environment
        args: Command-line arguments
        visualizer: Visualization object

    Returns:
        dict: Training statistics
    """
    print("\n" + "=" * 60)
    print("Training Actor-Critic Algorithm")
    print("=" * 60)
    print(f"Feature type: {args.feature_type}")
    print(f"GAE lambda: {args.gae_lambda}")

    # Initialize Actor-Critic agent
    ac_agent = ActorCriticAgent(
        env=env,
        feature_type=args.feature_type,  # Added: feature type parameter
        hidden_size=args.hidden_size,
        actor_lr=args.lr,
        critic_lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,  # Added: GAE lambda parameter
    )

    # Train the agent
    stats = ac_agent.train(
        n_episodes=args.n_episodes,
        max_steps=args.max_steps
    )

    # Get final policy
    policy = ac_agent.get_policy()

    visualizer.plot_ac_training_curves(
        episode_rewards=stats['episode_rewards'],
        actor_losses=stats.get('actor_losses', []),  # Use get to prevent KeyError
        critic_losses=stats.get('critic_losses', []),  # Use get to prevent KeyError
        title="Actor-Critic Training Curves",
        save=True
    )

    return {
        'policy': policy,
        'stats': stats
    }


def compare_algorithms(env, args, reinforce_results, ac_results, visualizer):
    """
    Compare REINFORCE and Actor-Critic algorithms

    Args:
        env: GridWorld environment
        args: Command-line arguments
        reinforce_results: REINFORCE results
        ac_results: Actor-Critic results
        visualizer: Visualization object
    """
    print("\n" + "=" * 60)
    print("Algorithm Comparison")
    print("=" * 60)
    print(f"Actor-Critic parameters: feature_type={args.feature_type}, gae_lambda={args.gae_lambda}")

    # 1. Policy comparison for both methods
    reinforce_policy = reinforce_results['policy']
    ac_policy = ac_results['policy']
    
    visualizer.plot_policy_comparison(
        env,
        reinforce_policy,
        ac_policy,
        "REINFORCE vs Actor-Critic: Policy Comparison"
    )

    # 2. Training rewards comparison for both methods
    visualizer.plot_algorithm_comparison(
        reinforce_results['stats']['episode_rewards'],
        ac_results['stats']['episode_rewards'],
        "REINFORCE vs Actor-Critic: Episode Rewards"
    )


def main():
    """Main function for Policy Gradient algorithm evaluation"""
    # Parse command-line arguments
    args = parse_arguments()

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Initialize Environment
    env = create_environment(args)

    print("=" * 60)
    print("Policy Gradient Algorithms Evaluation in GridWorld")
    print("=" * 60)
    print(f"Grid Size: {args.size}x{args.size}, Total States: {env.n_states}")
    print(f"Discount Factor (gamma): {args.gamma}")
    print(f"Action Space: {env.actions}")
    print(f"Target States: {args.target_states}")
    print(f"Forbidden States: {args.forbidden_states}")
    print(f"Reward Settings: boundary={args.r_bound}, forbidden={args.r_forbid}, "
          f"target={args.r_target}, default={args.r_default}")
    print(f"Training Episodes: {args.n_episodes}, Max Steps per Episode: {args.max_steps}")
    print(f"Feature Type: {args.feature_type}, GAE Lambda: {args.gae_lambda}")

    # Initialize Visualizer
    visualizer = GridWorldVisualizer(results_dir)

    # 2. Train REINFORCE Algorithm
    reinforce_results = train_reinforce(env, args, visualizer)

    # 3. Train Actor-Critic Algorithm
    ac_results = train_actor_critic(env, args, visualizer)

    # 4. Compare Algorithms
    compare_algorithms(env, args, reinforce_results, ac_results, visualizer)

    print("\n" + "=" * 60)
    print("Program execution completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()