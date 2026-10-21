import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class EstimatorVisualizer:
    """
    Estimator Visualizer - Visualization tool for gradient descent optimization results
    
    This class provides methods to visualize the trajectories and error convergence
    of various gradient descent algorithms (SGD, MBGD, BGD) in parameter space.
    """
    def __init__(self, environment, results, plot_samples=True):
        """
        Initialize the visualizer.
        
        Parameters:
        environment: Environment object containing data generation parameters.
        results: Dictionary containing results from different gradient descent methods.
        plot_samples: Boolean indicating whether to plot sample points (default: True).
        """
        self.env = environment
        self.results = results
        self.plot_samples = plot_samples
        self.samples = self.env.generate_samples() if plot_samples else None
        
        # Configuration for figure sizes
        self.fig_width = 16
        self.fig_height = 8
        
        # Font configuration
        plt.rcParams['font.family'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Colors for trajectories
        self.colors = ['blue', 'green', 'orange', 'purple', 'brown']
        
        # Line styles for error plots
        self.line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    def plot_all_results(self, save_path='.'):
        """
        Plot and save all results as 3 separate figures (one for each algorithm pair).
        
        Parameters:
        save_path: Directory path to save the figures (default: current directory).
        """
        # Create and save SGD figure
        self._plot_algorithm_pair('SGD', save_path)
        
        # Create and save MBGD figure
        self._plot_algorithm_pair('MBGD', save_path)
        
        # Create and save BGD figure
        self._plot_algorithm_pair('BGD', save_path)

    def _plot_algorithm_pair(self, mode, save_path):
        """
        Plot trajectory and error convergence for a specific algorithm.
        
        Parameters:
        mode: String indicating which algorithm to plot ('SGD', 'MBGD', 'BGD').
        save_path: Directory path to save the figure.
        """
        # Adjust subplot width ratio to give more space to the first subplot
        fig, axes = plt.subplots(1, 2, figsize=(self.fig_width, self.fig_height), gridspec_kw={'width_ratios': [1, 1]})
        
        # Plot trajectory on left subplot
        self._plot_trajectories(axes[0], mode=mode)
        
        # Plot error convergence on right subplot
        self._plot_errors(axes[1], mode=mode)
        
        plt.tight_layout()
        save_file = os.path.join(save_path, f'{mode}_results.png')
        plt.savefig(save_file, dpi=300)
        plt.show()
        plt.close(fig)

    def _plot_trajectories(self, ax, mode='SGD'):
        """
        Plot parameter space trajectories.
        
        Parameters:
        ax: Matplotlib axis object to plot on.
        mode: String indicating which algorithm to plot ('SGD', 'MBGD', 'BGD').
        """
        # Plot sample points if enabled
        if self.plot_samples and self.samples is not None:
            ax.scatter(self.samples[:, 0], self.samples[:, 1], 
                      alpha=0.3, s=5, label='Samples', color='lightgray', zorder=1)
        
        # Plot true mean (red star)
        ax.scatter(0, 0, color='red', s=100, marker='*', label='True Mean', zorder=5)
        
        # Draw boundary rectangle
        h = self.env.square_size / 2
        rect = patches.Rectangle((-h, -h), self.env.square_size, self.env.square_size,
                         fill=False, edgecolor='black', ls='--', linewidth=1)
        ax.add_patch(rect)
        
        color_idx = 0
        
        # Plot trajectories based on mode
        for method_name, method_data in self.results.items():
            if mode == 'SGD' and 'SGD' in method_name:
                traj = method_data['trajectory']
                ax.plot(traj[:, 0], traj[:, 1], label=method_name, 
                       color=self.colors[color_idx % len(self.colors)], lw=2, zorder=4)
                color_idx += 1
            elif mode == 'MBGD' and 'MBGD' in method_name:
                traj = method_data['trajectory']
                ax.plot(traj[:, 0], traj[:, 1], label=method_name, 
                       color=self.colors[color_idx % len(self.colors)], lw=2, zorder=4)
                color_idx += 1
            elif mode == 'BGD' and 'FullBatchGD' in method_name:
                traj = method_data['trajectory']
                ax.plot(traj[:, 0], traj[:, 1], label=method_name, 
                       color=self.colors[color_idx % len(self.colors)], lw=1.5, zorder=4)
                color_idx += 1
        
        # Set plot title and legend
        ax.set_title(f'{mode} Trajectories', fontweight='bold', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        
        # Set appropriate coordinate limits
        padding = 5
        all_traj_points = []
        for method_name, method_data in self.results.items():
            if ((mode == 'SGD' and 'SGD' in method_name) or
                (mode == 'MBGD' and 'MBGD' in method_name) or
                (mode == 'BGD' and 'FullBatchGD' in method_name)):
                all_traj_points.append(method_data['trajectory'])
        
        if all_traj_points:
            all_points = np.vstack(all_traj_points)
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
            
            x_center, y_center = (x_min + x_max)/2, (y_min + y_max)/2
            x_range = max(x_max - x_min, 20) + padding * 2
            y_range = max(y_max - y_min, 20) + padding * 2
            
            ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
            ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        else:
            ax.set_xlim(-20, 60)
            ax.set_ylim(-20, 60)
            
        # Set labels and grid
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_aspect('equal', adjustable='box')

    def _plot_errors(self, ax, mode='SGD'):
        """
        Plot error convergence over iterations.
        
        Parameters:
        ax: Matplotlib axis object to plot on.
        mode: String indicating which algorithm to plot ('SGD', 'MBGD', 'BGD').
        """
        iters = range(self.env.total_iterations + 1)
        color_idx = 0
        
        # Plot error convergence for each method
        for method_name, method_data in self.results.items():
            if mode == 'SGD' and 'SGD' in method_name:
                err = method_data['errors']
                ax.plot(iters, err, label=method_name, 
                       color=self.colors[color_idx % len(self.colors)], 
                       linewidth=1.5, 
                       linestyle=self.line_styles[color_idx % len(self.line_styles)])
                color_idx += 1
            elif mode == 'MBGD' and 'MBGD' in method_name:
                err = method_data['errors']
                ax.plot(iters, err, label=method_name, 
                       color=self.colors[color_idx % len(self.colors)], 
                       linewidth=1.5,
                       linestyle=self.line_styles[color_idx % len(self.line_styles)])
                color_idx += 1
            elif mode == 'BGD' and 'FullBatchGD' in method_name:
                err = method_data['errors']
                ax.plot(iters, err, label=method_name, 
                       color=self.colors[color_idx % len(self.colors)], 
                       linewidth=1.5,
                       linestyle=self.line_styles[color_idx % len(self.line_styles)])
                color_idx += 1
        
        # Set plot title and labels
        ax.set_title(f'{mode} Error Convergence', fontweight='bold', fontsize=12)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set appropriate y-axis limits
        all_errors = []
        for method_name, method_data in self.results.items():
            if ((mode == 'SGD' and 'SGD' in method_name) or
                (mode == 'MBGD' and 'MBGD' in method_name) or
                (mode == 'BGD' and 'FullBatchGD' in method_name)):
                all_errors.append(method_data['errors'])
        
        if all_errors:
            all_errors = np.concatenate(all_errors)
            ax.set_ylim(0, all_errors.max() * 1.1)