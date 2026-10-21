import argparse
from typing import List
import numpy as np
import os
import sys

# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Chapter3_Policy_and_Value_Iteration.src.algorithms.value_iteration import ValueIteration
from Chapter3_Policy_and_Value_Iteration.src.algorithms.policy_iteration import PolicyIteration
from Chapter3_Policy_and_Value_Iteration.src.algorithms.truncated_policy_iteration import TruncatedPI
from Chapter1_Basic_Concepts.src.environment_model import GridWorld
from Chapter3_Policy_and_Value_Iteration.src.visualization import ExperimentVisualizer

result_dir = os.path.join(os.path.dirname(current_dir), "results")

class Experiment:
    """GridWorld reinforcement learning experiment runner"""
    
    def __init__(self, env: GridWorld = None, theta: float = None, 
                 max_iteration: int = None, truncated_inner_iteration: int = None,
                 truncated_inner_iteration_list: List = None, result_dir: str = None):
        self.env = env
        self.v_star = None
        self.visualizer = ExperimentVisualizer(results_dir=result_dir)
        
        # Algorithm parameters
        self.theta = theta
        self.max_iteration = max_iteration
        self.truncated_inner_iteration = truncated_inner_iteration
        self.truncated_inner_iteration_list = truncated_inner_iteration_list
    
    def compute_optimal_value_function(self) -> np.ndarray:
        """Compute high-precision optimal value function as benchmark (V*)"""
        print("Computing optimal value function benchmark...")
        vi_solver = ValueIteration(self.env, self.theta, self.max_iteration)
        vi_solver.theta = 1e-12
        vi_solver.solve()
        self.v_star = vi_solver.V.copy()
        
        target = self.env.target_states[0] if self.env.target_states else 0
        print(f"Benchmark complete. Optimal value for target S{target}: {self.v_star[target]:.6f}")
        return self.v_star
    
    def run_comparison_experiment(self):
        """Compare value evolution at state S0 across algorithms"""
        print("\n" + "=" * 60)
        print("Experiment 1: Value Convergence at State S0")
        print("=" * 60)
        
        if self.v_star is None:
            self.compute_optimal_value_function()
        
        algorithms = {
            'Value Iteration': ValueIteration(self.env, self.theta, self.max_iteration),
            'Policy Iteration': PolicyIteration(self.env, self.theta, self.max_iteration),
            f'Truncated PI (inner={self.truncated_inner_iteration})': 
                TruncatedPI(self.env, self.theta, self.max_iteration, self.truncated_inner_iteration)
        }
        
        monitor_state = 0
        algorithms_results = {}
        
        for name, solver in algorithms.items():
            print(f"Running {name}...")
            solver.V = np.zeros(self.env.n_states)
            res = solver.solve()
            
            state_history = [v[monitor_state] for v in res['values']]
            algorithms_results[name] = (state_history, res['iterations'], res['time'])
            print(f"  -> Iterations: {res['iterations']}, Time: {res['time']:.4f}s")
        
        self.visualizer.plot_convergence_comparison(algorithms_results, self.v_star, monitor_state)
    
    def run_truncated_pi_experiment(self):
        """Analyze effect of truncation steps on convergence error"""
        print("\n" + "=" * 60)
        print("Experiment 2: Truncated PI Parameter Sensitivity")
        print("=" * 60)
        
        tpi_results = {}
        for x in self.truncated_inner_iteration_list:
            solver = TruncatedPI(self.env, x=x)
            solver.V = np.zeros(self.env.n_states)
            res = solver.solve()
            
            tpi_results[x] = (res['values'], len(res['values']), res['time'])
            error = np.max(np.abs(solver.V - self.v_star))
            print(f"TPI (x={x:3d}): Final error = {error:.2e}, Iterations = {len(res['values'])}")
        
        self.visualizer.plot_truncated_pi_sensitivity(tpi_results, self.v_star)
    
    def run_policy_comparison(self):
        """Compare optimal policies learned by different algorithms"""
        print("\n" + "=" * 60)
        print("Experiment 3: Optimal Policy Comparison")
        print("=" * 60)
        
        solvers = {
            'Value Iteration': ValueIteration(self.env, self.theta, self.max_iteration),
            'Policy Iteration': PolicyIteration(self.env, self.theta, self.max_iteration),
            f'Truncated PI (inner={self.truncated_inner_iteration})': 
                TruncatedPI(self.env, self.theta, self.max_iteration, self.truncated_inner_iteration)
        }
        
        solvers_results = {}
        for name, solver in solvers.items():
            print(f"Solving {name}...")
            res = solver.solve()
            solvers_results[name] = (solver, res['iterations'])
            print(f"  -> Iterations: {res['iterations']}")
        
        self.visualizer.plot_policy_comparison(self.env, solvers_results)


def parse_arguments():
    """Parse command line arguments"""
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
    parser.add_argument("--r_forbid", type=float, default=-1, help="Forbidden state reward")
    parser.add_argument("--r_target", type=float, default=1, help="Target state reward")
    parser.add_argument("--r_default", type=float, default=0, help="Default reward")
    
    # Algorithm parameters
    parser.add_argument("--theta", type=float, default=1e-6, help="Convergence threshold")
    parser.add_argument("--max_iteration", type=int, default=1000, help="Max iterations")
    parser.add_argument("--truncated_inner_iteration", type=int, default=20, 
                       help="Inner iterations for Truncated PI")
    parser.add_argument("--truncated_inner_iteration_list", nargs='+', type=int, 
                       default=[1, 5, 10, 20, 50, 100], help="TPI parameter test values")
    
    return parser.parse_args()


def main():
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
    
    # Run experiments
    exp = Experiment(env, args.theta, args.max_iteration,
                    args.truncated_inner_iteration, args.truncated_inner_iteration_list, result_dir)
    
    exp.run_comparison_experiment()
    exp.run_truncated_pi_experiment()
    exp.run_policy_comparison()


if __name__ == "__main__":
    main()