# visualization.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

class ExperimentVisualizer:
    """Visualization class for Dynamic Programming algorithms experiments"""
    
    def __init__(self, results_dir='results'):
        # Initialize all variables
        self.results_dir = os.path.join(current_dir, results_dir)
        
        # Define fixed colors and line styles for convergence comparison
        self.convergence_colors = ['blue', 'red', 'green']
        self.convergence_styles = ['-', '--', '-.']
        
        # Define colors and markers for TPI sensitivity analysis
        self.tpi_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        self.tpi_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        
        # Define color mapping for grid world visualization
        self.grid_colors = {'normal': 'white', 'forbidden': 'orange', 'target': 'lightblue'}
        
        # Define arrow symbols for policy visualization
        self.arrow_symbols = {'up': '↑', 'right': '→', 'down': '↓', 'left': '←', 'stay': '○'}
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
    
    def plot_convergence_comparison(self, algorithms_results, v_star, monitor_state=0):
        """Experiment 1: Compare the value evolution curves of the three algorithms at a specific state"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (name, (state_history, iterations, _)) in enumerate(algorithms_results.items()):
            color = self.convergence_colors[i % len(self.convergence_colors)]
            style = self.convergence_styles[i % len(self.convergence_styles)]
            
            ax.plot(range(len(state_history)), state_history,
                     label=f"{name} ({iterations} iter)",
                     color=color, linestyle=style, linewidth=2)
        
        # Plot the V* reference line
        ax.axhline(y=v_star[monitor_state], color='black', linestyle=':', label='Optimal $V^*(S0)$')
        
        ax.set_xlabel('Outer Iterations')
        ax.set_ylabel(f'Value Function $V(S{monitor_state})$')
        ax.set_title('Convergence Comparison at State S0')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save the figure
        fig_path = os.path.join(self.results_dir, 'convergence_comparison_S0.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {fig_path}")
        plt.show()
    
    def plot_truncated_pi_sensitivity(self, tpi_results, v_star):
        """Experiment 2: Analyze the effect of truncation step x on convergence error"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (x, (values_history, iterations, _)) in enumerate(tpi_results.items()):
            errors = [np.max(np.abs(v - v_star)) + 1e-15 for v in values_history]
            
            color = self.tpi_colors[i % len(self.tpi_colors)]
            marker = self.tpi_markers[i % len(self.tpi_markers)]
            
            ax.semilogy(range(len(errors)), errors,
                         label=f'x={x} ({len(errors)} iter)',
                         color=color, marker=marker,
                         markevery=max(1, len(errors)//10), markersize=6)
        
        ax.set_xlabel('Outer Iterations')
        ax.set_ylabel('Max Error $||V - V^*||_\infty$ (Log Scale)')
        ax.set_title('Effect of Evaluation Steps (x) on TPI Convergence')
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        
        # Save the figure
        fig_path = os.path.join(self.results_dir, 'TPI_error_vs_x.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {fig_path}")
        plt.show()
    
    def visualize_grid_world(self, env, policy, value_function=None, title="", ax=None):
        """Visualization logic for GridWorld environment"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
        
        # Draw grid lines
        for i in range(env.size + 1):
            ax.axhline(i, color='black', linewidth=1)
            ax.axvline(i, color='black', linewidth=1)
        
        for state in range(env.n_states):
            row = state // env.size
            col = state % env.size
            x_center = col + 0.5
            y_center = env.size - row - 0.5  # Map to the drawing coordinate system
            
            # Color fill
            if state in env.target_states:
                facecolor = self.grid_colors['target']
            elif state in env.forbidden_states:
                facecolor = self.grid_colors['forbidden']
            else:
                facecolor = self.grid_colors['normal']
            
            rect = patches.Rectangle((col, env.size - row - 1), 1, 1,
                                   facecolor=facecolor, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            
            # State number
            ax.text(x_center, y_center, str(state), ha='center', va='center',
                    fontsize=10, color='black', weight='bold')
            
            # Value function display
            if value_function is not None:
                ax.text(x_center, y_center + 0.3, f'{value_function[state]:.2f}',
                        ha='center', va='center', fontsize=8, color='blue')
            
            # Policy arrow logic
            action_probs = policy[state] if policy.ndim > 1 else np.eye(env.n_actions)[policy[state]]
            max_prob = np.max(action_probs)
            max_actions = [env.actions[i] for i in range(env.n_actions) if action_probs[i] == max_prob]
                
            if max_actions:
                action_symbol = self.arrow_symbols[max_actions[0]]
                ax.text(x_center, y_center - 0.2, action_symbol,
                        ha='center', va='center', fontsize=16, color='black', weight='bold')
    
        ax.set_xlim(0, env.size)
        ax.set_ylim(0, env.size)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if ax is None:
            plt.show()
    
    def plot_policy_comparison(self, env, solvers_results):
        """Experiment 3: Visual comparison of optimal policies from different algorithms"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        for i, (name, (solver, iterations)) in enumerate(solvers_results.items()):
            # Call the visualization method
            self.visualize_grid_world(
                env=env,
                policy=solver.policy,
                value_function=solver.V,
                title=f"{name}\n({iterations} iters)",
                ax=axes[i]
            )
        
        plt.tight_layout()
        # Save the figure
        fig_path = os.path.join(self.results_dir, 'policy_comparison.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {fig_path}")
        plt.show()