import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import argparse

# Get absolute path of current file (main.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get parent directory path
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Add parent directory to Python path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Chapter1_Basic_Concepts.src.environment_model import GridWorld
from algorithms.bellman_equation import BellmanSolver
from Chapter2_Bellman_Equations.src.visualization import GridWorldVisualizer


def create_policies_5x5(grid_world, random_seed=None):
    policies = []
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Policy 1: Good deterministic policy
    policy1 = np.zeros((grid_world.n_states, grid_world.n_actions))
    p1_actions = {
        0: 'right', 1: 'right', 2: 'right', 3: 'down', 4: 'down',
        5: 'up', 6: 'up', 7: 'right', 8: 'down', 9: 'down',
        10: 'up', 11: 'left', 12: 'down', 13: 'right', 14: 'down',
        15: 'up', 16: 'right', 17: 'stay', 18: 'left', 19: 'down',
        20: 'up', 21: 'right', 22: 'up', 23: 'left', 24: 'left'
    }
    for s, a in p1_actions.items():
        policy1[s, grid_world.actions.index(a)] = 1.0
    policies.append(('Policy 1 - A Good Policy', policy1))
    
    # Policy 2: Another good deterministic policy
    policy2 = np.zeros((grid_world.n_states, grid_world.n_actions))
    p2_actions = {
        0: 'right', 1: 'right', 2: 'right', 3: 'right', 4: 'down',
        5: 'up', 6: 'up', 7: 'right', 8: 'right', 9: 'down',
        10: 'up', 11: 'left', 12: 'down', 13: 'right', 14: 'down',
        15: 'up', 16: 'right', 17: 'stay', 18: 'left', 19: 'down',
        20: 'up', 21: 'right', 22: 'up', 23: 'left', 24: 'left'
    }
    for s, a in p2_actions.items():
        policy2[s, grid_world.actions.index(a)] = 1.0
    policies.append(('Policy 2 - Another Good Policy', policy2))

    # Policy 3: Poor policy (always move right)
    policy3 = np.zeros((grid_world.n_states, grid_world.n_actions))
    for s in range(grid_world.n_states):
        policy3[s, grid_world.actions.index('right')] = 1.0
    policies.append(('Policy 3 - Poor Policy (Always Right)', policy3))

    # Policy 4: Random Policy
    policy4 = np.zeros((grid_world.n_states, grid_world.n_actions))
    n_states = grid_world.n_states
    n_actions = grid_world.n_actions
    for s in range(n_states):
        random_probs = np.random.rand(n_actions)
        policy4[s, :] = random_probs / random_probs.sum()
    policies.append(('Policy 4 - Random Policy (Equal Chance)', policy4))
    
    return policies


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Grid World Environment with Random Policy")
    
    parser.add_argument("--size", type=int, default=5, help="Grid size")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    
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

    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize environment
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
    
    visualizer = GridWorldVisualizer()
    
    solver = BellmanSolver(env)
    test_policies = create_policies_5x5(env)

    # Create result directory
    result_dir = os.path.join(os.path.dirname(current_dir), "results")
    os.makedirs(result_dir, exist_ok=True)

    save_path_1 = os.path.join(result_dir, "grid_world_policy_comparison_closed.png")
    save_path_2 = os.path.join(result_dir, "grid_world_policy_comparison_iterative.png")
    results = {}
    print(f"Starting Bellman equation computation for {len(test_policies)} policies...\n")
    
    # Analyze test policies using closed-form solution
    for name, p_matrix in test_policies:
        print(f"[Closed-Form Solution] Computing value function for policy: '{name}'...")
        v_sol = solver.closed_form_solution(p_matrix)
        results[name] = {'value': v_sol}
        print(f"  Policy '{name}' computation completed successfully")

    visualizer.visualize_results(env, test_policies, results, save_path_1)
    print("\n[Results] All policy value functions have been computed and visualized.")

    # Analyze test policies using iterative solution
    for name, p_matrix in test_policies:
        print(f"[Iterative Solution] Computing value function for policy: '{name}'...")
        v_sol = solver.iterative_solution(p_matrix)
        results[name] = {'value': v_sol}
        print(f"  Policy '{name}' computation completed successfully")

    visualizer.visualize_results(env, test_policies, results, save_path_2)
    print("\n[Results] All policy value functions have been computed and visualized successfully.")

if __name__ == "__main__":
    main()