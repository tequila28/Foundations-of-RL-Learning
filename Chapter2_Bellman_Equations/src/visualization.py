import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


class GridWorldVisualizer:
    def __init__(self):
        """
        Initialize Grid World Visualizer
        """
        # Define action arrow symbols
        self.arrows = {
            'up': '↑',
            'right': '→',
            'down': '↓',
            'left': '←',
            'stay': '○'
        }
        
        # Define colors for different state types
        self.colors = {
            'normal': 'white',
            'forbidden': 'orange',
            'target': 'lightblue',
            'boundary': 'lightgray'
        }
        
        # Default figure sizes
        self.figsize_single = (10, 8)
        self.figsize_comparison_width = 5
        self.figsize_comparison_height = 6
    
    def visualize_grid_world(self, env, policy, value_function=None, title="", ax=None):
        """
        Visualize a single grid world
        
        Parameters:
        -----------
        env : GridWorld
            Grid world environment
        policy : np.ndarray
            Policy matrix
        value_function : np.ndarray, optional
            Value function
        title : str
            Title for the plot
        ax : matplotlib.axes.Axes, optional
            Matplotlib axes object
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize_single)
        else:
            fig = ax.get_figure()
        
        # Create grid lines
        for i in range(env.size + 1):
            ax.axhline(i, color='black', linewidth=1)
            ax.axvline(i, color='black', linewidth=1)
        
        for state in range(env.n_states):
            row = state // env.size
            col = state % env.size
            x_center = col + 0.5
            y_center = env.size - row - 0.5
            
            # Determine state type and color
            if state in env.target_states:
                facecolor = self.colors['target']
            elif state in env.forbidden_states:
                facecolor = self.colors['forbidden']
            else:
                facecolor = self.colors['normal']
            
            # Draw state cell
            rect = patches.Rectangle(
                (col, env.size - row - 1), 1, 1,
                facecolor=facecolor, edgecolor='black', alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add state number
            ax.text(
                x_center, y_center, str(state),
                ha='center', va='center',
                fontsize=10, color='black', weight='bold'
            )
            
            # Add value function if provided
            if value_function is not None:
                ax.text(
                    x_center, y_center + 0.3,
                    f'{value_function[state]:.2f}',
                    ha='center', va='center',
                    fontsize=8, color='blue'
                )
            
            # Add action symbol
            action_probs = policy[state]
            max_prob = np.max(action_probs)
            max_actions = [env.actions[i] for i in range(env.n_actions) if action_probs[i] == max_prob]
                
            if max_actions:
                action_symbol = self.arrows[max_actions[0]]
                ax.text(
                    x_center, y_center - 0.2,
                    action_symbol,
                    ha='center', va='center',
                    fontsize=16, color='black', weight='bold'
                )
        
        # Set plot properties
        ax.set_xlim(0, env.size)
        ax.set_ylim(0, env.size)
        ax.set_aspect('equal')
        ax.set_title(title if title else 'Grid World', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        return fig
    
    def visualize_results(self, grid_world, policies, results, save_path=None):
        """
        Generate comparison visualizations for multiple policies (Horizontal Layout)
        
        Parameters:
        -----------
        grid_world : GridWorld
            Grid world environment
        policies : list
            List of policies, each element is a tuple (policy_name, policy_matrix)
        results : dict
            Dictionary containing results for each policy
        save_path : str, optional
            Path to save the visualization image
        """
        num_policies = len(policies)
        
        # Create subplots: 1 row, num_policies columns
        fig, axes = plt.subplots(
            1, num_policies,
            figsize=(self.figsize_comparison_width * num_policies, self.figsize_comparison_height)
        )
        
        # Handle case when there's only one policy
        if num_policies == 1:
            axes = [axes]
        
        for idx, (name, policy) in enumerate(policies):
            ax = axes[idx]
            
            # Get value function for this policy
            value_function = results[name]['value'] if name in results else None
            
            # Visualize grid world with current subplot
            self.visualize_grid_world(grid_world, policy, value_function, title=name, ax=ax)
            
            # Remove legend for subplots to save space
            if ax.get_legend():
                ax.get_legend().remove()
        
        plt.tight_layout()
        
        # Save image if save_path is provided
        if save_path:
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
            os.makedirs(dir_path, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Image saved to: {save_path}")
        
        plt.show()
        
        return fig

