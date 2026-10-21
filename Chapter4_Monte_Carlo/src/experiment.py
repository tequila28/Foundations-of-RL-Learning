import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# Environment path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(os.path.dirname(parent_dir))

# Ensure the results directory exists
results_dir = os.path.join(os.path.dirname(current_dir), 'results')

# Import the three MC algorithms
from Chapter4_Monte_Carlo.src.algorithms.mc_basic import MCBasic
from Chapter4_Monte_Carlo.src.algorithms.mc_exploring_starts import MCExploringStarts
from Chapter4_Monte_Carlo.src.algorithms.mc_epsilon_greedy import MCEpsilonGreedy
from Chapter4_Monte_Carlo.src.visualization import GridVisualizer
from Chapter1_Basic_Concepts.src.environment_model import GridWorld


def parse_arguments():
    parser = argparse.ArgumentParser(description="GridWorld RL Experiments")
    
    # Environment parameters
    parser.add_argument("--size", type=int, default=5, help="Grid size")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    
    # State configuration
    parser.add_argument("--forbidden_states", nargs='+', type=int, 
                       default=[6, 7, 12, 16, 18, 21], help="Forbidden states")
    parser.add_argument("--target_states", nargs='+', type=int, 
                       default=[17], help="Target states")
    
    # Reward settings
    parser.add_argument("--r_bound", type=float, default=-1, help="Boundary reward")
    parser.add_argument("--r_forbid", type=float, default=-10, help="Forbidden state reward")
    parser.add_argument("--r_target", type=float, default=1, help="Target state reward")
    parser.add_argument("--r_default", type=float, default=0, help="Default reward")
    
    # MC Basic parameters
    parser.add_argument("--mc_basic_episode_length", type=int, default=100, 
                       help="MC Basic episode length")
    parser.add_argument("--mc_basic_iterations", type=int, default=1000, 
                       help="MC Basic training iterations")
    
    # MC Exploring Starts parameters
    parser.add_argument("--mc_es_episode_length", type=int, default=100, 
                       help="MC Exploring Starts episode length")
    parser.add_argument("--mc_es_iterations", type=int, default=10000, 
                       help="MC Exploring Starts training iterations")
    
    # MC ε-greedy (first) parameters
    parser.add_argument("--mc_eps1_episode_length", type=int, default=1000, 
                       help="MC ε-greedy (first) episode length")
    parser.add_argument("--mc_eps1_epsilon", type=float, default=0.1, 
                       help="MC ε-greedy (first) epsilon value")
    parser.add_argument("--mc_eps1_iterations", type=int, default=1000, 
                       help="MC ε-greedy (first) training iterations")
    
    # MC ε-greedy (second) parameters
    parser.add_argument("--mc_eps2_episode_length", type=int, default=1000, 
                       help="MC ε-greedy (second) episode length")
    parser.add_argument("--mc_eps2_epsilon", type=float, default=0.2, 
                       help="MC ε-greedy (second) epsilon value")
    parser.add_argument("--mc_eps2_iterations", type=int, default=1000, 
                       help="MC ε-greedy (second) training iterations")
    
    # scatter parameters
    parser.add_argument("--mc_episode_length", type=int, default=100000, 
                       help="MC ε-greedy episode length")
    parser.add_argument("--mc_eps_epsilon1", type=float, default=1.0, 
                       help="MC ε-greedy epsilon value")
    parser.add_argument("--mc_eps_epsilon2", type=float, default=0.2, 
                       help="MC ε-greedy epsilon value")
    
    return parser.parse_args()



def main(args):
    
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
    
    visualizer = GridVisualizer(results_dir)
    
    # Additional visualization: Epsilon exploration analysis
    print("\n" + "="*60)
    print("Running Epsilon Exploration Analysis")
    print("="*60)
    
    epsilon_values = [args.mc_eps_epsilon1, args.mc_eps_epsilon2]
    episode_lengths = [args.mc_episode_length]
    
    visualizer.visualize_epsilon_exploration(
        env=env,
        epsilon_values=epsilon_values,
        episode_lengths=episode_lengths,
    )
    
    # 2. Define algorithms to compare
    algorithms = [
        {
            "name": "MC Basic",
            "agent_class": MCBasic,
            "init_params": {"episode_length": args.mc_basic_episode_length},
            "train_params": {"num_iterations": args.mc_basic_iterations},
            "color": "blue"
        },
        {
            "name": "MC Exploring Starts",
            "agent_class": MCExploringStarts,
            "init_params": {"episode_length": args.mc_es_episode_length},
            "train_params": {"num_iterations": args.mc_es_iterations},
            "color": "green"
        },
        {
            "name": f"MC ε-greedy (ε={args.mc_eps1_epsilon})",
            "agent_class": MCEpsilonGreedy,
            "init_params": {"episode_length": args.mc_eps1_episode_length, 
                           "epsilon": args.mc_eps1_epsilon},
            "train_params": {"num_iterations": args.mc_eps1_iterations},
            "color": "red"
        },
        {
            "name": f"MC ε-greedy (ε={args.mc_eps2_epsilon})",
            "agent_class": MCEpsilonGreedy,
            "init_params": {"episode_length": args.mc_eps2_episode_length, 
                           "epsilon": args.mc_eps2_epsilon},
            "train_params": {"num_iterations": args.mc_eps2_iterations},
            "color": "orange"
        }
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  
    
    all_results = []
    
    for idx, algo_config in enumerate(algorithms):
        print(f"\n{'='*60}")
        print(f"Training {algo_config['name']}")
        print(f"{'='*60}")
        
        # Create agent with initialization parameters
        agent = algo_config["agent_class"](env, **algo_config["init_params"])
        
        print(f"Training for {algo_config['train_params']['num_iterations']} iterations...")
        agent.train(**algo_config["train_params"])
        
        V, policy = agent.get_results()
        
        all_results.append({
            "name": algo_config["name"],
            "agent": agent,
            "V": V,
            "policy": policy,
            "color": algo_config["color"]
        })
        
        if idx < 4: 
            visualizer.visualize_grid_world(
                env, 
                policy, 
                value_function=V, 
                title=f"{algo_config['name']}\nFinal Policy",
                ax=axes[idx]
            )
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'mc_policy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("Algorithm Performance Summary")
    print("="*60)
    
    print(f"\n{'Algorithm':<25} {'Avg |V|':<12} {'Max V':<10} {'Min V':<10}")
    print("-" * 60)
    
    for result in all_results:
        V = result["V"]
        avg_value = np.mean(np.abs(V))
        max_value = np.max(V)
        min_value = np.min(V)
        
        print(f"{result['name']:<25} {avg_value:<12.4f} {max_value:<10.4f} {min_value:<10.4f}")
    
    print("-" * 60)
    
    return all_results

if __name__ == "__main__":
    args = parse_arguments()
    results = main(args)