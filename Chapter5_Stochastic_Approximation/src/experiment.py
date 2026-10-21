import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
import os

# Environment path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(os.path.dirname(parent_dir))

# Ensure the results directory exists
results_dir = os.path.join(parent_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

from Chapter5_Stochastic_Approximation.src.algorithms.bgd import BGD
from Chapter5_Stochastic_Approximation.src.algorithms.mbgd import MBGD
from Chapter5_Stochastic_Approximation.src.algorithms.sgd import SGD
from Chapter5_Stochastic_Approximation.src.environment import Environment
from Chapter5_Stochastic_Approximation.src.visualization import EstimatorVisualizer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Mean Estimation Experiments with Different Gradient Descent Methods")
    
    # Environment parameters
    parser.add_argument("--square_size", type=float, default=30.0, 
                       help="Size of the square for uniform distribution")
    parser.add_argument("--num_samples", type=int, default=400, 
                       help="Number of samples to generate")
    parser.add_argument("--total_iterations", type=int, default=400, 
                       help="Total number of iterations for optimization")
    
    # Initial point parameters
    parser.add_argument("--init_x", type=float, default=50.0, 
                       help="Initial x-coordinate")
    parser.add_argument("--init_y", type=float, default=50.0, 
                       help="Initial y-coordinate")
    
    # Optimization parameters
    parser.add_argument("--batch_size", type=int, default=[1, 10, 50], nargs='+',
                       help="Batch size(s) for mini-batch gradient descent (can specify multiple)")
    parser.add_argument("--constant_alpha", type=float, default=0.005, 
                       help="Constant learning rate (used when alpha_type is 'constant')")
    
    return parser.parse_args()


def analyze_computational_cost(total_iters, batch_sizes):
    print("\n" + "="*50)
    print("Computational Cost Analysis")
    print("="*50)
    
    # Include standard batch sizes
    all_batches = sorted(batch_sizes)
    
    for m in all_batches:
        print(f"Batch m={m}: Total samples processed = {m * total_iters:,}")

def main():
    print("🚀 Starting Mean Estimation Experiment...")
    
    # Parse arguments
    args = parse_arguments()
    
    # Ensure batch_size is a list
    if isinstance(args.batch_size, int):
        args.batch_size = [args.batch_size]
    
    # Print parameters
    print(f"\n📊 Parameters:")
    print(f"  Square Size: {args.square_size}")
    print(f"  Number of Samples: {args.num_samples}")
    print(f"  Total Iterations: {args.total_iterations}")
    print(f"  Initial Point: [{args.init_x}, {args.init_y}]")
    print(f"  Batch Sizes: {args.batch_size}")
    
    # Initialize environment
    env = Environment(
        square_size=args.square_size,
        num_samples=args.num_samples,
        total_iterations=args.total_iterations
    )
    
    env.generate_samples()
    # Initialize optimizers
    bgd = BGD(env)
    sgd = SGD(env)
    
    # Run different algorithms
    print("\n🔬 Running algorithms...")
    
    results = {}
    
    # Run each algorithm twice: 1/k and constant
    learning_rate_configs = [
        ('1/k', '1/k', None),  # name, type, constant value
        ('constant', 'constant', args.constant_alpha)
    ]
    
    # SGD (Batch=1)
    for lr_name, lr_type, lr_value in learning_rate_configs:
        sgd_traj, sgd_err = sgd.estimate(
            initial_point=np.array([args.init_x, args.init_y]),
            alpha_type=lr_type,
            constant_alpha=lr_value
        )
        results[f'SGD (m=1, α={lr_name})'] = {'trajectory': sgd_traj, 'errors': sgd_err}
    
    # MBGD for each batch size
    for batch_size in args.batch_size:
        mbgd = MBGD(env)
        for lr_name, lr_type, lr_value in learning_rate_configs:
            mbgd_traj, mbgd_err = mbgd.estimate(
                initial_point=np.array([args.init_x, args.init_y]),
                batch_size=batch_size,
                alpha_type=lr_type,
                constant_alpha=lr_value
            )
            results[f'MBGD (m={batch_size}, α={lr_name})'] = {'trajectory': mbgd_traj, 'errors': mbgd_err}
    
    # Full batch GD
    for lr_name, lr_type, lr_value in learning_rate_configs:
        bgd_traj, bgd_err = bgd.estimate(
            initial_point=np.array([args.init_x, args.init_y]),
            alpha_type=lr_type,
            constant_alpha=lr_value
        )
        results[f'FullBatchGD (α={lr_name})'] = {'trajectory': bgd_traj, 'errors': bgd_err}
    
    # Visualization
    print("🎨 Generating plots...")
    visualizer = EstimatorVisualizer(env, results)
    visualizer.plot_all_results(results_dir)
    
    # Computational cost analysis
    analyze_computational_cost(args.total_iterations, args.batch_size)
    
    # Final results
    print("\n📈 Final Results:")
    for method_name, method_data in results.items():
        traj = method_data['trajectory']
        err = method_data['errors']
        print(f"{method_name}: Final Estimate: {traj[-1].round(3)}, Final Error: {err[-1]:.6f}")
    
    # Add learning rate comparison analysis
    print("\n" + "="*50)
    print("Learning Rate Comparison")
    print("="*50)
    
    # Group comparison by algorithm type
    algorithm_types = {
        'SGD': [],
        'MBGD': [],
        'FullBatchGD': []
    }
    
    for method_name in results.keys():
        if 'SGD' in method_name and 'MBGD' not in method_name and 'FullBatchGD' not in method_name:
            algorithm_types['SGD'].append(method_name)
        elif 'MBGD' in method_name:
            algorithm_types['MBGD'].append(method_name)
        elif 'FullBatchGD' in method_name:
            algorithm_types['FullBatchGD'].append(method_name)
    
    for algo_name, methods in algorithm_types.items():
        if methods:
            print(f"\n{algo_name} Methods:")
            for method in sorted(methods):
                err = results[method]['errors'][-1]
                traj = results[method]['trajectory'][-1]
                print(f"  {method}: Error={err:.6f}, Final Point={traj}")
    
    print(f"\n✅ Done. Results saved to '{results_dir}'")

if __name__ == "__main__":
    main()