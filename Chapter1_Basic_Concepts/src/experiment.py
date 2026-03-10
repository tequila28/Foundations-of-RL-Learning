import numpy as np
import os
import argparse
from environment_model import GridWorld
from visualization import GridWorldVisualizer

## Get absolute path of current file (experiment.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get parent directory path
parent_dir = os.path.dirname(current_dir)
# Add parent directory to Python path

def generate_random_policy(env: GridWorld, seed: int = None) -> np.ndarray:
    """
    Generate a random policy where each state has random action probabilities.
    
    Parameters:
    -----------
    env : GridWorld
        The grid world environment
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    np.ndarray
        Random policy matrix of shape (n_states, n_actions)
    """
    if seed is not None:
        np.random.seed(seed)
    policy = np.random.random((env.n_states, env.n_actions))
    # Normalize to get probability distribution
    policy = policy / policy.sum(axis=1, keepdims=True)
    return policy


def print_environment_info(env: GridWorld):
    """
    Print comprehensive information about the environment.
    
    Parameters:
    -----------
    env : GridWorld
        The grid world environment
    """
    print("=" * 60)
    print("GRID WORLD ENVIRONMENT INFORMATION")
    print("=" * 60)
    print(f"Grid Size: {env.size} x {env.size}")
    print(f"Total States: {env.n_states}")
    print(f"Discount Factor (γ): {env.gamma}")
    print(f"Available Actions: {env.actions}")
    print(f"Number of Actions: {env.n_actions}")
    print(f"Forbidden States: {env.forbidden_states}")
    print(f"Target States: {env.target_states}")
    print(f"Reward Values - Boundary: {env.r_bound}, Forbidden: {env.r_forbid}, "
          f"Target: {env.r_target}, Default: {env.r_default}")
    print("=" * 60)


def print_policy_info(env: GridWorld, policy: np.ndarray):
    """
    Print information about the policy.
    
    Parameters:
    -----------
    env : GridWorld
        The grid world environment
    policy : np.ndarray
        Policy matrix
    """
    print("\nPOLICY INFORMATION")
    print("=" * 40)
    
    for state in range(env.n_states):
        action_probs = policy[state]
        max_prob = np.max(action_probs)
        optimal_actions = [env.actions[i] for i in range(env.n_actions) if action_probs[i] == max_prob]
        
        state_type = "Target" if state in env.target_states else \
                   "Forbidden" if state in env.forbidden_states else "Normal"
        
        print(f"State {state:2d} ({state_type}): ", end="")
        for i, action in enumerate(env.actions):
            print(f"{action}: {action_probs[i]:.2f}  ", end="")
        print(f"→ Optimal: {optimal_actions}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Grid World Environment with Random Policy")
    
    parser.add_argument("--size", type=int, default=5, help="Grid size (default: 5)")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor (default: 0.9)")
    parser.add_argument("--actions", nargs='+', default=['up', 'right', 'down', 'left', 'stay'],
                       help="Available actions")
    
    parser.add_argument("--forbidden_states", nargs='+', type=int, default=[6, 7, 12, 16, 18, 21],
                       help="Forbidden states")
    parser.add_argument("--target_states", nargs='+', type=int, default=[17],
                       help="Target states")

    parser.add_argument("--r_bound", type=float, default=-1, help="Boundary reward")
    parser.add_argument("--r_forbid", type=float, default=-1, help="Forbidden state reward")
    parser.add_argument("--r_target", type=float, default=1, help="Target state reward")
    parser.add_argument("--r_default", type=float, default=0, help="Default reward")
    
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: None)")
    
    return parser.parse_args()


def main():
    """
    Main function to demonstrate the grid world environment with random policy.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize environment
    env = GridWorld(
        size=args.size,
        gamma=args.gamma,
        actions=args.actions,
        forbidden_states=args.forbidden_states,
        target_states=args.target_states,
        r_bound=args.r_bound,
        r_forbid=args.r_forbid,
        r_target=args.r_target,
        r_default=args.r_default
    )
    
    # Generate random policy
    policy = generate_random_policy(env, seed=args.seed)
    
    # Print environment and policy information
    print_environment_info(env)
    print_policy_info(env, policy)
    
    # Create result directory
    result_dir = os.path.join(parent_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    
    # Visualize the environment with policy and save to file
    save_path_1 = os.path.join(result_dir, "grid_world_policy.png")
    print(f"\nSaving visualization to: {save_path_1}")
    visualizer = GridWorldVisualizer()
    visualizer.visualize(env, policy, save_path=save_path_1)
    
    # Compute and display value function for the random policy
    print("\nCOMPUTING VALUE FUNCTION FOR RANDOM POLICY...")
    P_pi, r_pi = env.get_policy_matrices(policy)
    value_function = np.linalg.solve(np.eye(env.n_states) - env.gamma * P_pi, r_pi)
    
    print("\nVALUE FUNCTION:")
    for state in range(env.n_states):
        print(f"State {state:2d}: {value_function[state]:.4f}")
    
    # Visualize with value function and save to file
    save_path_2 = os.path.join(result_dir, "grid_world_policy_value.png")
    print(f"\nSaving visualization with state value to: {save_path_2}")
    visualizer.visualize(env, policy, value_function, save_path=save_path_2)
    
    print("\nAll visualizations have been saved to the 'result' directory.")


if __name__ == "__main__":
    main()