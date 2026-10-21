import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from environment_model import GridWorld


class GridWorldVisualizer:
    def __init__(self):
        """
        Initialize the GridWorldVisualizer with default settings.
        """
        # Define action arrows
        self.arrows = {
            'up': '↑',
            'right': '→',
            'down': '↓',
            'left': '←',
            'stay': '○'
        }
        
        # Color mapping for different state types
        self.colors = {
            'normal': 'white',
            'forbidden': 'orange',
            'target': 'lightblue',
            'boundary': 'lightgray'
        }
    
    def visualize(self, env: GridWorld, policy, value_function=None, save_path=None):
        """
        Visualize the grid world environment with policy actions and optional value function.
        
        Parameters:
        -----------
        env : GridWorld
            The grid world environment
        policy : array-like
            Policy matrix of shape (n_states, n_actions)
        value_function : array-like, optional
            Value function for each state
        save_path : str, optional
            Path to save the visualization image. If None, image is not saved
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create grid
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
            rect = patches.Rectangle((col, env.size - row - 1), 1, 1,
                                    facecolor=facecolor, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            
            # Add state number
            ax.text(x_center, y_center, str(state),
                    ha='center', va='center', fontsize=10, color='black', weight='bold')
            
            # Add value function if provided
            if value_function is not None:
                ax.text(x_center, y_center + 0.3, f'{value_function[state]:.2f}',
                        ha='center', va='center', fontsize=8, color='blue')
            
            # Add policy action
            action_probs = policy[state]
            max_prob = max(action_probs)
            max_action_idx = action_probs.index(max_prob) if hasattr(action_probs, 'index') else action_probs.argmax()
            action_symbol = self.arrows[env.actions[max_action_idx]]
            ax.text(x_center, y_center - 0.2, action_symbol,
                    ha='center', va='center', fontsize=16, color='black', weight='bold')
        
        # Set plot properties
        ax.set_xlim(0, env.size)
        ax.set_ylim(0, env.size)
        ax.set_aspect('equal')
        
        # Determine title
        if value_function is not None:
            ax.set_title('Grid World Environment with Policy and State Value', fontsize=14)
        else:
            ax.set_title('Grid World Environment with Policy', fontsize=14)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Create legend
        legend_elements = [
            patches.Patch(facecolor=self.colors['normal'], label='Normal State', edgecolor='black'),
            patches.Patch(facecolor=self.colors['forbidden'], label='Forbidden State', edgecolor='black'),
            patches.Patch(facecolor=self.colors['target'], label='Target State', edgecolor='black'),
            plt.Line2D([0], [0], color='black', label='Stay Action (○)', markersize=8, linestyle='None'),
            plt.Line2D([0], [0], color='white', label='Up (↑)', markersize=0, linestyle='None'),
            plt.Line2D([0], [0], color='white', label='Right (→)', markersize=0, linestyle='None'),
            plt.Line2D([0], [0], color='white', label='Down (↓)', markersize=0, linestyle='None'),
            plt.Line2D([0], [0], color='white', label='Left (←)', markersize=0, linestyle='None')
        ]
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5))
        
        plt.tight_layout()
        
        # Save the plot if save_path is provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        

