import argparse
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Environment path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(os.path.dirname(parent_dir))

from Chapter1_Basic_Concepts.src.environment_model import GridWorld
from Chapter6_Temporal_Difference.src.algorithms.off_policy_qlearning import OffPolicyQLearning
from Chapter6_Temporal_Difference.src.algorithms.on_policy_qlearning import OnPolicyQLearning
from Chapter6_Temporal_Difference.src.algorithms.n_step_sarsa import NStepSARSA
from Chapter6_Temporal_Difference.src.algorithms.sarsa import SARSA
from Chapter6_Temporal_Difference.src.algorithms.expected_sarsa import ExpectedSARSA
from Chapter6_Temporal_Difference.src.visualization import AlgorithmVisualizer

# Ensure the results directory exists
results_dir = os.path.join(parent_dir, 'results')
os.makedirs(results_dir, exist_ok=True)


def train_algorithm(agent, num_episodes: int = 1000, max_steps: int = 100) -> dict:
    """
    Train a single algorithm
    
    Args:
        agent: Algorithm instance
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        
    Returns:
        Training history dictionary
    """
    algorithm_name = agent.__class__.__name__
    print(f"Training {algorithm_name}...")
    
    # Enable history tracking
    agent.track_history = True
    
    # Train the algorithm
    Q, history = agent.train(
        num_episodes=num_episodes,
        max_steps=max_steps,
        verbose=False
    )
    
    if not history:
        history = {
            'episode_rewards': [],
            'episode_steps': [],
            'td_errors': []
        }
    
    history['agent'] = agent
    history['algorithm_name'] = algorithm_name
    
    return history


def train_all_algorithms(env, num_episodes: int = 1000, max_steps: int = 100, 
                         learning_rate: float = 0.1, epsilon: float = 0.2,
                         epsilon_decay: float = 0.998, epsilon_min: float = 0.01,
                         gamma: float = 0.95, n_steps: list = [1, 10]) -> dict:
    """
    Train all TD algorithms
    
    Args:
        env: GridWorld environment
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        learning_rate: Learning rate for all algorithms
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay rate
        epsilon_min: Minimum exploration rate
        gamma: Discount factor
        n_steps: List of n values for N-step SARSA
        
    Returns:
        Dictionary containing all algorithm results
    """
    print("\n" + "="*60)
    print("TRAINING ALL TD ALGORITHMS")
    print("="*60)
    
    all_results = {}
    
    # 1. Off-policy Q-Learning
    print("\n1. Training Off-policy Q-Learning...")
    qlearning_agent = OffPolicyQLearning(
        env=env,
        learning_rate=learning_rate,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        gamma=gamma,
        track_history=True
    )
    qlearning_history = train_algorithm(qlearning_agent, num_episodes, max_steps)
    all_results['off_policy_qlearning'] = {
        'agent': qlearning_agent,
        'history': qlearning_history,
        'display_name': 'Off-policy Q-Learning'
    }
    
    # 2. On-policy Q-Learning
    print("2. Training On-policy Q-Learning...")
    onpolicy_agent = OnPolicyQLearning(
        env=env,
        learning_rate=learning_rate,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        gamma=gamma,
        track_history=True
    )
    onpolicy_history = train_algorithm(onpolicy_agent, num_episodes, max_steps)
    all_results['on_policy_qlearning'] = {
        'agent': onpolicy_agent,
        'history': onpolicy_history,
        'display_name': 'On-policy Q-Learning'
    }
    
    # 3. SARSA
    print("3. Training SARSA...")
    sarsa_agent = SARSA(
        env=env,
        learning_rate=learning_rate,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        gamma=gamma,
        track_history=True
    )
    sarsa_history = train_algorithm(sarsa_agent, num_episodes, max_steps)
    all_results['sarsa'] = {
        'agent': sarsa_agent,
        'history': sarsa_history,
        'display_name': 'SARSA'
    }
    
    # 4. Expected SARSA
    print("4. Training Expected SARSA...")
    expected_sarsa_agent = ExpectedSARSA(
        env=env,
        learning_rate=learning_rate,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        gamma=gamma,
        track_history=True
    )
    expected_sarsa_history = train_algorithm(expected_sarsa_agent, num_episodes, max_steps)
    all_results['expected_sarsa'] = {
        'agent': expected_sarsa_agent,
        'history': expected_sarsa_history,
        'display_name': 'Expected SARSA'
    }
    
    # 5. N-step SARSA
    for i, n in enumerate(n_steps, 1):
        print(f"5.{i}. Training N-step SARSA (n={n})...")
        n_step_sarsa_agent = NStepSARSA(
            env=env,
            n=n,
            learning_rate=learning_rate,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            gamma=gamma,
            track_history=True
        )
        n_step_sarsa_history = train_algorithm(n_step_sarsa_agent, num_episodes, max_steps)
        all_results[f'n_step_sarsa_{n}'] = {
            'agent': n_step_sarsa_agent,
            'history': n_step_sarsa_history,
            'display_name': f'N-step SARSA (n={n})'
        }
    
    print("\n✓ All algorithms trained successfully!")
    return all_results


def visualize_training_results(all_results: dict, visualizer: AlgorithmVisualizer) -> None:
    """
    Visualize training results
    
    Args:
        all_results: Training results for all algorithms
        visualizer: Visualizer object
    """
    print("\n" + "="*60)
    print("VISUALIZING TRAINING RESULTS")
    print("="*60)
    
    # 1. Extract training histories for comparison
    print("\n1. Training curves comparison...")
    training_histories = {}
    for algo_key, result in all_results.items():
        training_histories[algo_key] = result['history']
    
    visualizer.compare_multiple_algorithms(training_histories, window_size=20)
    
    # 2. Show final learned policies
    print("\n2. Final learned policies...")
    n_algorithms = len(all_results)
    n_cols = 3
    n_rows = (n_algorithms + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    for idx, (algo_key, result) in enumerate(all_results.items()):
        if idx >= len(axes):
            break
            
        agent = result['agent']
        display_name = result['display_name']
        
        try:
            policy = agent.get_greedy_policy()
            value_function = np.max(agent.Q, axis=1)
            eps = getattr(agent, 'epsilon', 0)
            title = f"{display_name}\nε={eps:.4f}"
            
            # Visualize
            visualizer.visualize_grid_world(
                policy=policy,
                value_function=value_function,
                title=title,
                ax=axes[idx],
            )
            
        except Exception as e:
            print(f"Error visualizing {algo_key}: {e}")
            axes[idx].text(0.5, 0.5, f"Error: {e}", 
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(display_name)
    
    # Hide unused subplots
    for idx in range(len(all_results), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Final Policies Learned by Different TD Algorithms', fontsize=16, y=1.02)
    plt.savefig(os.path.join(results_dir, 'final_policies_comparison.png'), dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def print_training_summary(all_results: dict) -> None:
    """
    Print training summary
    
    Args:
        all_results: Training results for all algorithms
    """
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    print(f"\n{'Algorithm':<30} {'Best Reward':<12} {'Avg Last 10':<12} {'Avg Last 50':<12} {'Final ε':<10}")
    print("-" * 80)
    
    for algo_key, result in all_results.items():
        history = result['history']
        display_name = result['display_name']
        
        if 'episode_rewards' in history and history['episode_rewards']:
            rewards = history['episode_rewards']
            
            # Calculate metrics
            best_reward = np.max(rewards)
            avg_last_10 = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
            avg_last_50 = np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards)
            
            # Get final epsilon
            if 'epsilon_history' in history and history['epsilon_history']:
                final_eps = history['epsilon_history'][-1]
            else:
                final_eps = getattr(result['agent'], 'epsilon', 0)
            
            print(f"{display_name:<30} "
                  f"{best_reward:<12.2f} "
                  f"{avg_last_10:<12.2f} "
                  f"{avg_last_50:<12.2f} "
                  f"{final_eps:<10.4f}")
        else:
            print(f"{display_name:<30} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}")


def parse_arguments():
    """
    Parse command-line arguments for the GridWorld RL experiments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="GridWorld RL Experiments")
    
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
    parser.add_argument("--r_forbid", type=float, default=-10, 
                       help="Forbidden state reward (default: -10)")
    parser.add_argument("--r_target", type=float, default=1, 
                       help="Target state reward (default: 1)")
    parser.add_argument("--r_default", type=float, default=0, 
                       help="Default reward (default: 0)")
    parser.add_argument("--gamma", type=float, default=0.9, 
                       help="Discount factor (default: 0.9)")
    
    # Training parameters
    parser.add_argument("--num_episodes", type=int, default=100, 
                       help="Number of training episodes (default: 500)")
    parser.add_argument("--max_steps", type=int, default=1000, 
                       help="Maximum steps per episode (default: 100)")
    
    # Algorithm hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.1, 
                       help="Learning rate (default: 0.1)")
    parser.add_argument("--epsilon", type=float, default=0.2, 
                       help="Initial exploration rate (default: 0.2)")
    parser.add_argument("--epsilon_decay", type=float, default=0.998, 
                       help="Epsilon decay rate (default: 0.998)")
    parser.add_argument("--epsilon_min", type=float, default=0.01, 
                       help="Minimum exploration rate (default: 0.01)")
    
    # N-step SARSA parameters
    parser.add_argument("--n_steps", nargs='+', type=int, default=[1, 10], 
                       help="N values for N-step SARSA (default: 1 10)")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # 1. Create environment
    print("\n" + "="*60)
    print("INITIALIZING GRIDWORLD ENVIRONMENT")
    print("="*60)
    env = GridWorld(
        size=args.size,
        forbidden_states=args.forbidden_states,
        target_states=args.target_states,
        r_bound=args.r_bound,
        r_forbid=args.r_forbid,
        r_target=args.r_target,
        r_default=args.r_default
    )
    print(f"Grid size: {args.size}x{args.size}")
    print(f"Total states: {env.n_states}")
    print(f"Target states: {args.target_states}")
    print(f"Forbidden states: {args.forbidden_states}")
    print(f"Rewards - Boundary: {args.r_bound}, Forbidden: {args.r_forbid}, "
          f"Target: {args.r_target}, Default: {args.r_default}")
    print(f"Discount factor (γ): {args.gamma}")
    
    # 2. Create visualizer
    visualizer = AlgorithmVisualizer(env)
    
    # 3. Train all algorithms
    all_results = train_all_algorithms(
        env=env,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        gamma=args.gamma,
        n_steps=args.n_steps
    )
    
    # 4. Print training summary
    print_training_summary(all_results)
    
    visualize_training_results(all_results, visualizer)
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
   

if __name__ == "__main__":
    main()