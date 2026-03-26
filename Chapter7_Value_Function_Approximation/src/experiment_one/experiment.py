# main.py
import numpy as np
import os
import sys
import argparse

# Environment path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(parent_dir)))

# Ensure the results directory exists
results_dir = os.path.join(os.path.dirname(parent_dir), 'experiment_one_results')
os.makedirs(results_dir, exist_ok=True)

# Import required modules
from Chapter1_Basic_Concepts.src.environment_model import GridWorld
from Chapter7_Value_Function_Approximation.src.experiment_one.algorithms.policy_evaluator import EpisodeGenerator
from Chapter7_Value_Function_Approximation.src.experiment_one.algorithms.bellman_iteration import BellmanIteration
from Chapter7_Value_Function_Approximation.src.experiment_one.algorithms.td_linear import TDLinear
from Chapter7_Value_Function_Approximation.src.experiment_one.visualization import GridWorldVisualizer


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
    parser.add_argument("--r_forbid", type=float, default=-1, 
                       help="Forbidden state reward (default: -1)")
    parser.add_argument("--r_target", type=float, default=1, 
                       help="Target state reward (default: 1)")
    parser.add_argument("--r_default", type=float, default=0, 
                       help="Default reward (default: 0)")
    parser.add_argument("--gamma", type=float, default=0.9, 
                       help="Discount factor (default: 0.9)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed (default: 42)")
    
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


def main():
    """Main function for TD-Linear algorithm evaluation"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # 1. Initialize Environment
    env = create_environment(args)
    
    print("=" * 60)
    print("TD-Linear Algorithm Evaluation in GridWorld")
    print("=" * 60)
    print(f"Grid Size: {args.size}x{args.size}, Total States: {env.n_states}")
    print(f"Discount Factor (gamma): {args.gamma}")
    print(f"Action Space: {env.actions}")
    print(f"Target States: {args.target_states}")
    print(f"Forbidden States: {args.forbidden_states}")
    print(f"Reward Settings: boundary={args.r_bound}, forbidden={args.r_forbid}, "
          f"target={args.r_target}, default={args.r_default}")
    
    # 2. Calculate Ground Truth
    print("\n" + "=" * 60)
    print("[Step 1] Compute Ground Truth (Bellman Iteration)...")
    print("=" * 60)
    bellman = BellmanIteration(env)
    V_ground_truth = bellman.iterate(max_iterations=1000, threshold=1e-6)
    
    # Initialize Visualizer
    visualizer = GridWorldVisualizer(results_dir)
    
    # Print Ground Truth
    visualizer.print_state_values(env, V_ground_truth, "Ground Truth State Values")
    
    # 3. Generate Training Data
    print("\n" + "=" * 60)
    print("[Step 2] Generate Training Data...")
    print("=" * 60)
    episode_generator = EpisodeGenerator(env, seed=args.seed)
    episodes = []
    n_episodes = 500
    
    for _ in range(n_episodes):
        episode = episode_generator.generate_episode(max_steps=500)
        episodes.append(episode)
    
    print(f"Generated {n_episodes} episodes, each with maximum 500 steps")
    
    # 4. Train TD-Linear Algorithm
    print("\n" + "=" * 60)
    print("[Step 3] Train TD-Linear Algorithm...")
    print("=" * 60)
    poly_V_dict = {}
    learning_rates = {}
    rmse_results = {1: [], 2: [], 3: []}  # Store RMSE for each order
    
    for order in [1, 2, 3]:
        print(f"\n--- Training with Polynomial Features (Order={order}) ---")
        alpha = 0.0005
        learning_rates[order] = alpha
        td_agent = TDLinear(env, order, alpha=alpha)
        
        # Training process
        for i, episode in enumerate(episodes):
            # Learning rate decay
            if i > 0 and i % 100 == 0:
                td_agent.alpha *= 0.99
                
            # Process episode
            for state, action_idx, reward in episode:
                # Get action name
                action = env.actions[action_idx]
                
                # Get next state
                next_state, _ = env.transition_logic(state, action)
                
                # Update TD algorithm
                td_agent.update(state, next_state, reward)
            
            # Calculate RMSE after each episode
            V_estimate = []
            for s in range(env.n_states):
                val = td_agent.value_estimate(s)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                V_estimate.append(val)
            
            V_estimate = np.array(V_estimate)
            error_diff = V_estimate - V_ground_truth
            rmse = np.sqrt(np.mean(error_diff ** 2))
            rmse_results[order].append(rmse)
        
        # Get final estimated values
        V_estimate = []
        for s in range(env.n_states):
            val = td_agent.value_estimate(s)
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            V_estimate.append(val)
        
        V_estimate = np.array(V_estimate)
        poly_V_dict[order] = V_estimate
        
        # Print results
        visualizer.print_state_values(env, V_estimate, f"Order {order} State Values")
        
        # Calculate final RMSE
        error_diff = V_estimate - V_ground_truth
        rmse = np.sqrt(np.mean(error_diff ** 2))
        print(f"Final RMSE: {rmse:.4f}")
    
    # 5. Visualization
    print("\n" + "=" * 60)
    print("[Step 4] Visualization...")
    print("=" * 60)
    
    # 5.1 Plot Ground Truth
    print("\nPlotting Ground Truth 3D Graph...")
    visualizer.plot_ground_truth_3d(env, V_ground_truth)
    
    # 5.2 Plot TD-Linear Results
    print("\nPlotting 3D State Value Graphs for Different Polynomial Orders...")
    visualizer.plot_3d_values_multi(env, poly_V_dict, 
                         "TD-Linear Algorithm: State Value Estimates with Polynomial Features")
    
    # 5.3 Plot RMSE Curves
    print("\nPlotting RMSE Curves vs. Episode...")
    visualizer.plot_rmse_curves_multi(rmse_results, learning_rates)
    
    # 5.4 Plot Comparison Summary
    print("\nPlotting Comparison Summary...")
    visualizer.plot_comparison_summary(env, V_ground_truth, poly_V_dict)
    
    # 6. Final Summary
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    
    print("\nTarget State(17) Value Comparison:")
    print(f"Ground Truth: {V_ground_truth[17]:.6f}")
    
    for order in [1, 2, 3]:
        V_est = poly_V_dict[order]
        diff = V_est[17] - V_ground_truth[17]
        relative_error = abs(diff / V_ground_truth[17]) * 100 if V_ground_truth[17] != 0 else 0
        print(f"Order {order}: {V_est[17]:.6f} (Error: {diff:+.6f}, Relative Error: {relative_error:.2f}%)")
    
    print("\nAverage Value Comparison Across All States:")
    print(f"Ground Truth Average: {np.mean(V_ground_truth):.6f}")
    
    for order in [1, 2, 3]:
        V_est = poly_V_dict[order]
        mean_est = np.mean(V_est)
        diff = mean_est - np.mean(V_ground_truth)
        relative_error = abs(diff / np.mean(V_ground_truth)) * 100 if np.mean(V_ground_truth) != 0 else 0
        print(f"Order {order} Average: {mean_est:.6f} (Error: {diff:+.6f}, Relative Error: {relative_error:.2f}%)")
    
    # Print RMSE statistics
    print("\nRMSE Statistics:")
    for order in [1, 2, 3]:
        final_rmse = rmse_results[order][-1] if rmse_results[order] else 0
        initial_rmse = rmse_results[order][0] if rmse_results[order] else 0
        improvement = ((initial_rmse - final_rmse) / initial_rmse * 100) if initial_rmse != 0 else 0
        print(f"Order {order}: Initial RMSE: {initial_rmse:.4f}, Final RMSE: {final_rmse:.4f}, "
              f"Improvement: {improvement:.2f}%")
    
    print("\n" + "=" * 60)
    print("Program execution completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()