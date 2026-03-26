import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional
import os
from Chapter1_Basic_Concepts.src.environment_model import GridWorld


class GridWorldVisualizer:
    """Visualization class for GridWorld reinforcement learning experiments."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory to save visualization results
        """
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
    
    def plot_rmse_curves_multi(self, rmse_results: Dict[int, List[float]], 
                               learning_rates: Dict[int, float] = None,
                               title: str = "TD-Linear Algorithm: RMSE Learning Curves") -> None:
        """
        Plot multiple RMSE curves on a single figure.
        
        Args:
            rmse_results: Dictionary with polynomial order as keys and RMSE lists as values
            learning_rates: Dictionary of learning rates for each order
            title: Main figure title
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, order in enumerate([1, 2, 3]):
            ax = axes[idx]
            rmse_values = rmse_results[order]
            
            if not rmse_values:
                continue
                
            # Plot original curve
            episodes_indices = range(1, len(rmse_values) + 1)
            ax.plot(episodes_indices, rmse_values, linewidth=1, color='blue', 
                    alpha=0.5, label='Raw RMSE')
            
            # Plot smoothed curve
            window_size = 20
            if len(rmse_values) >= window_size:
                rmse_smooth = np.convolve(rmse_values, np.ones(window_size)/window_size, 
                                         mode='valid')
                episodes_smooth = episodes_indices[window_size-1:]
                ax.plot(episodes_smooth, rmse_smooth, linewidth=2, color='red', 
                       alpha=0.8, label='Smoothed RMSE')
            
            # Configure axes
            ax.set_xlabel('Episode Index', fontsize=10)
            ax.set_ylabel('RMSE', fontsize=10)
            
            # Add title with information
            title_text = f'Polynomial Order = {order}'
            if learning_rates and order in learning_rates:
                title_text += f'\nLearning Rate: {learning_rates[order]:.6f}'
            title_text += f'\nFinal RMSE: {rmse_values[-1]:.4f}'
            
            ax.set_title(title_text, fontsize=11, pad=20)
            ax.grid(True, alpha=0.3)
            
            # Add text information
            info_text = f'Initial RMSE: {rmse_values[0]:.4f}\nFinal RMSE: {rmse_values[-1]:.4f}'
            ax.text(0.5, 0.95, info_text, transform=ax.transAxes, 
                   ha='center', va='top', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.legend(fontsize=8, loc='upper right')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(self.results_dir, f"rmse_curves_{len(rmse_values)}_episodes.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_3d_values_multi(self, env: GridWorld, V_dict: Dict[int, np.ndarray], 
                            title: str = "TD-Linear Algorithm: State Value Estimates") -> None:
        """
        Plot 3D surface plots for state values with different polynomial orders.
        
        Args:
            env: GridWorld environment
            V_dict: Dictionary with polynomial order as keys and state value arrays as values
            title: Main figure title
        """
        fig = plt.figure(figsize=(15, 5))
        
        for idx, order in enumerate([1, 2, 3], 1):
            ax = fig.add_subplot(1, 3, idx, projection='3d')
            V = V_dict[order]
            
            # Create grid
            X, Y = np.meshgrid(range(env.size), range(env.size))
            Z = V.reshape(env.size, env.size)
            
            # Transpose Z matrix to match coordinate system
            Z = Z.T
            
            # Plot 3D surface
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            
            # Configure axes
            ax.set_xlabel('Row', fontsize=10, labelpad=10)
            ax.set_ylabel('Column', fontsize=10, labelpad=10)
            ax.set_zlabel('State Value V(s)', fontsize=10, labelpad=10)
            
            # Set axis limits and ticks
            ax.set_xlim(0, env.size-1)
            ax.set_ylim(0, env.size-1)
            ax.set_xticks(range(env.size))
            ax.set_yticks(range(env.size))
            
            # Add value range information
            if not np.isnan(Z).all():
                z_min, z_max = np.nanmin(Z), np.nanmax(Z)
                z_range = f"V(s) ∈ [{z_min:.3f}, {z_max:.3f}]"
            else:
                z_range = "V(s) = NaN"
            
            ax.set_title(f'Polynomial Order = {order}\n{z_range}', 
                        fontsize=12, fontweight='bold')
            
            # Set view angle
            ax.view_init(elev=30, azim=-45)
            
            # Add color bar
            if not np.isnan(Z).all():
                fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(self.results_dir, "3d_state_values.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_ground_truth_3d(self, env: GridWorld, V_ground_truth: np.ndarray) -> None:
        """
        Plot 3D surface plot for ground truth state values.
        
        Args:
            env: GridWorld environment
            V_ground_truth: Ground truth state value array
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create grid
        X, Y = np.meshgrid(range(env.size), range(env.size))
        Z = V_ground_truth.reshape(env.size, env.size)
        
        # Transpose Z matrix
        Z = Z.T
        
        # Plot 3D surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        # Configure axes
        ax.set_xlabel('Row', labelpad=10)
        ax.set_ylabel('Column', labelpad=10)
        ax.set_zlabel('State Value V(s)', labelpad=10)
        
        # Set axis limits and ticks
        ax.set_xlim(0, env.size-1)
        ax.set_ylim(0, env.size-1)
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        
        # Add title
        ax.set_title('Ground Truth - Bellman Iteration Results', 
                    fontsize=14, fontweight='bold')
        
        # Set view angle
        ax.view_init(elev=30, azim=-45)
        
        # Add color bar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(self.results_dir, "ground_truth_3d.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_comparison_summary(self, env: GridWorld, 
                               V_ground_truth: np.ndarray,
                               V_dict: Dict[int, np.ndarray]) -> None:
        """
        Plot a comprehensive comparison of all polynomial orders.
        
        Args:
            env: GridWorld environment
            V_ground_truth: Ground truth state values
            V_dict: Dictionary of estimated state values for each polynomial order
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Plot 1: Ground Truth
        ax1 = axes[0]
        V_matrix = V_ground_truth.reshape(env.size, env.size)
        im1 = ax1.imshow(V_matrix, cmap='viridis', aspect='equal')
        ax1.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Add value annotations
        for i in range(env.size):
            for j in range(env.size):
                val = V_matrix[i, j]
                ax1.text(j, i, f'{val:.2f}', ha='center', va='center', 
                        color='white' if val < np.median(V_matrix) else 'black',
                        fontsize=8)
        
        # Plot 2-4: Polynomial order comparisons
        for idx, order in enumerate([1, 2, 3], 1):
            ax = axes[idx]
            V = V_dict[order]
            V_matrix = V.reshape(env.size, env.size)
            im = ax.imshow(V_matrix, cmap='viridis', aspect='equal')
            
            # Calculate error
            error = V - V_ground_truth
            rmse = np.sqrt(np.mean(error ** 2))
            mae = np.mean(np.abs(error))
            
            ax.set_title(f'Order {order}\nRMSE: {rmse:.4f}, MAE: {mae:.4f}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row' if idx == 1 else '')
            
            plt.colorbar(im, ax=ax, shrink=0.8)
            
            # Add value annotations
            for i in range(env.size):
                for j in range(env.size):
                    val = V_matrix[i, j]
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                           color='white' if val < np.median(V_matrix) else 'black',
                           fontsize=8)
        
        plt.suptitle('State Value Comparison: Ground Truth vs. TD-Linear Estimates', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(self.results_dir, "comparison_summary.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_state_values(self, env: GridWorld, V: np.ndarray, 
                          title: str = "State Values") -> None:
        """
        Print state values in a formatted table.
        
        Args:
            env: GridWorld environment
            V: State value array
            title: Table title
        """
        print(f"\n{title}:")
        for i in range(env.size):
            row_str = ""
            for j in range(env.size):
                state = i * env.size + j
                value = V[state]
                row_str += f"{value:8.4f} "
            print(f"Row {i}: {row_str}")
        
        # Mark special states
        print("\nSpecial States:")
        for state in env.target_states:
            row, col = state // env.size, state % env.size
            print(f"  Target State {state} (Row {row}, Column {col}): {V[state]:.4f}")
        
        for state in env.forbidden_states:
            row, col = state // env.size, state % env.size
            print(f"  Forbidden State {state} (Row {row}, Column {col}): {V[state]:.4f}")
        
        # Print statistics
        print(f"\nStatistics:")
        print(f"  Minimum: {np.min(V):.4f}")
        print(f"  Maximum: {np.max(V):.4f}")
        print(f"  Mean: {np.mean(V):.4f}")
        print(f"  Standard Deviation: {np.std(V):.4f}")
    
    