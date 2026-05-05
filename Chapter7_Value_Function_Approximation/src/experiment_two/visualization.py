import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Optional, Union, Any
from Chapter1_Basic_Concepts.src.environment_model import GridWorld

# Environment path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(parent_dir)))

from Chapter1_Basic_Concepts.src.environment_model import GridWorld
results_dir = os.path.join(os.path.dirname(parent_dir), 'experiment_two_results')

class AlgorithmVisualizer:
    """Unified visualizer class supporting multiple TD learning algorithms"""
    
    def __init__(self, env: GridWorld):
        """
        Initialize the visualizer
        
        Args:
            env: GridWorld environment instance
        """
        # Initialize instance variables
        self.env = env
        
        # Color mapping for different algorithms
        self.algorithm_colors = {
            'on_policy_qlearning_fa': 'green',
            'sarsa_fa': 'orange',
        }
        
        # Display names for algorithms
        self.algorithm_names = {
            'onpolicy_agent': 'On-policy Q-Learning',
            'sarsa_agent': 'SARSA',
        
        }
        
        # Cell type colors (moved from visualize_grid_world)
        self.cell_colors = {
            'normal': 'white',
            'forbidden': 'orange',
            'target': 'lightblue'
        }
        
        # Arrow symbols for actions
        self.arrow_symbols = {
            'up': '↑',
            'right': '→',
            'down': '↓',
            'left': '←',
            'stay': '○'
        }
    
    def visualize_grid_world(self, 
                             policy: np.ndarray, 
                             value_function: Optional[np.ndarray] = None, 
                             title: str = "", 
                             ax: Optional[plt.Axes] = None) -> Tuple[Optional[plt.Figure], plt.Axes]:
        """
        Visualize GridWorld environment (consistent with GridVisualizer class)
        
        Args:
            policy: Policy matrix (n_states, n_actions) or action index array
            value_function: Value function for each state
            title: Plot title
            ax: Matplotlib axis
            
        Returns:
            fig: Figure object (returns None if ax is provided, otherwise figure)
            ax: Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
        else:
            fig = ax.figure
        
        for state in range(self.env.n_states):
            row = state // self.env.size
            col = state % self.env.size
            x_center = col + 0.5
            y_center = self.env.size - row - 0.5
            
            # 1. Determine fill color
            if state in self.env.target_states:
                facecolor = self.cell_colors['target']
            elif state in self.env.forbidden_states:
                facecolor = self.cell_colors['forbidden']
            else:
                facecolor = self.cell_colors['normal']
            
            # 2. Draw cell
            rect = patches.Rectangle((col, self.env.size - row - 1), 1, 1,
                                facecolor=facecolor, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            
            # 3. State number (simplified, only display number)
            ax.text(x_center, y_center, str(state), ha='center', va='center',
                    fontsize=10, color='black', weight='bold')
            
            # 4. Value function display
            if value_function is not None:
                ax.text(x_center, y_center + 0.3, f'{value_function[state]:.2f}',
                        ha='center', va='center', fontsize=8, color='blue')
            
            # 5. Policy arrow logic
            if policy is not None:
                if policy.ndim > 1:
                    action_probs = policy[state]
                else:
                    action_probs = np.eye(self.env.n_actions)[policy[state]]
                    
                max_prob = np.max(action_probs)
                
                # Find actions with highest probability
                max_action_indices = [i for i, prob in enumerate(action_probs) if prob == max_prob]
                
                if max_action_indices:
                    # Use only the first found best action (consistent with GridVisualizer)
                    action_name = self.env.actions[max_action_indices[0]]
                    action_symbol = self.arrow_symbols.get(action_name, '?')
                    ax.text(x_center, y_center - 0.2, action_symbol,
                            ha='center', va='center', fontsize=16, 
                            color='black', weight='bold')
        
        # Final formatting
        ax.set_xlim(0, self.env.size)
        ax.set_ylim(0, self.env.size)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
        return fig, ax
    
    def compare_multiple_algorithms(self, 
                                   algorithm_histories: Dict[str, Dict],
                                   window_size: int = 20,
                                   figsize: Tuple[int, int] = (25, 10)) -> None:
        """
        Compare training histories of multiple algorithms
        
        Args:
            algorithm_histories: Algorithm history dictionary {algorithm_name: history_data}
            window_size: Moving average window size
            figsize: Figure size
        """
        n_algorithms = len(algorithm_histories)
        if n_algorithms == 0:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes = axes.flatten()
        
        metrics_config = [
            ('episode_rewards', 'Total Reward per Episode', 'Reward', True),
            ('td_errors', 'TD Error Convergence', 'TD Error', False),
        ]
        
        def moving_average(data, window):
            """Calculate moving average"""
            if len(data) < window:
                return np.array(data)
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Plot each metric
        for i, (metric, title, ylabel, show_raw) in enumerate(metrics_config[:5]):
            ax = axes[i]
            
            for algo_name, history in algorithm_histories.items():
                if metric not in history:
                    continue
                    
                data = history[metric]
                print(algo_name)
                color = self.algorithm_colors.get(algo_name, 'gray')
                label = self.algorithm_names.get(algo_name, algo_name)
                
                if show_raw and len(data) > 0:
                    # Plot raw data (low opacity)
                    ax.plot(data, '-', color=color, alpha=0.2, linewidth=0.5)
                
                # Plot moving average
                if len(data) >= window_size:
                    ma_data = moving_average(data, window_size)
                    ax.plot(range(window_size-1, len(data)), ma_data, '-', 
                           color=color, label=label, linewidth=2, alpha=0.8)
                else:
                    ax.plot(data, '-', color=color, label=label, linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Episode')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            if i == 0:  # Show legend only in the first subplot
                ax.legend(fontsize=8, loc='best')
        
        plt.suptitle('Multi-Algorithm Training Comparison', fontsize=16, y=1.0)
        plt.savefig(os.path.join(results_dir, 'multi_algorithm_comparison.png'), dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()