import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Optional
import os

from Chapter4_Monte_Carlo.src.algorithms.mc_epsilon_greedy import MCEpsilonGreedy


class GridVisualizer:
    def __init__(self, result_dir: Optional[str] = None):
        """
        Initialize the GridVisualizer with necessary configurations.
        
        Parameters:
        -----------
        result_dir : str, optional
            Directory to save visualization results. If None, results won't be saved.
        """
        # Initialize all defined values
        self.result_dir = result_dir
        self.arrows = {'up': '↑', 'right': '→', 'down': '↓', 'left': '←', 'stay': '○'}
        self.colors = {'normal': 'white', 'forbidden': 'orange', 'target': 'lightblue'}
        
        # Create result directory if specified
        if self.result_dir is not None:
            os.makedirs(self.result_dir, exist_ok=True)
        
        # Set fonts to support symbols
        plt.rcParams['font.family'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def visualize_epsilon_exploration(self, env, epsilon_values, episode_lengths, num_episodes: int = 5):
        """
        Visualize agent trajectories and state-action visit frequencies for different epsilon values
        
        Parameters:
        -----------
        env : GridWorld
            The GridWorld environment
        epsilon_values : list
            List of epsilon values to test
        episode_lengths : list
            List of episode lengths to test
        num_episodes : int
            Number of episodes to generate (default: 5)
        """
        
        # Create figure for visit frequency scatter plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Store all data for scatter plot
        all_x_values = []  # State-action pair indices
        all_y_values = []  # Visit counts
        all_colors = []    # Color by epsilon value
        all_markers = []   # Marker by episode length
        all_labels = []    # Labels for data points
        
        # Create colormap for epsilon values
        eps_colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(epsilon_values)))
        eps_color_map = {eps: eps_colors[i] for i, eps in enumerate(epsilon_values)}
        
        # Marker styles for episode lengths
        length_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        marker_map = {len_val: length_markers[i % len(length_markers)] 
                    for i, len_val in enumerate(episode_lengths)}
        
        for eps_idx, epsilon in enumerate(epsilon_values):
            for len_idx, episode_length in enumerate(episode_lengths):
                # Create agent
                agent = MCEpsilonGreedy(env, episode_length=episode_length, epsilon=epsilon)
                
                # Track state-action visit counts
                sa_visit_counts = np.zeros((env.n_states, env.n_actions))
                
                # Generate episodes and track visits
                for _ in range(num_episodes):
                    episode = agent.generate_episode()
                    
                    # Record state-action visits
                    for state, action, _ in episode:
                        sa_visit_counts[state, action] += 1
                
                # Process and collect visit data
                for s in range(env.n_states):
                    for a in range(env.n_actions):
                        visit_count = sa_visit_counts[s, a]
                        if visit_count > 0:
                            # Create state-action index
                            sa_index = s * env.n_actions + a
                            
                            all_x_values.append(sa_index)
                            all_y_values.append(visit_count)
                            all_colors.append(eps_color_map[epsilon])
                            all_markers.append(marker_map[episode_length])
                            all_labels.append(f'(s{s},a{a}:{env.actions[a]})\nε={epsilon},len={episode_length}')
        
        # Create scatter plot
        if all_x_values:
            # Group points by marker for legend
            unique_markers = set(all_markers)
            for marker in unique_markers:
                # Find indices for this marker
                indices = [i for i, m in enumerate(all_markers) if m == marker]
                
                # Get corresponding episode length
                sample_idx = indices[0]
                sample_label_idx = all_labels[sample_idx].split('\n')[-1]  # Get "ε=...,len=..." part
                marker_label = sample_label_idx.split(',')[-1]  # Get "len=..." part
                
                # Extract just the length value
                length_value = marker_label.replace('len=', '')
                
                # Filter data for this marker
                x_vals = [all_x_values[i] for i in indices]
                y_vals = [all_y_values[i] for i in indices]
                colors = [all_colors[i] for i in indices]
                
                # Plot with this marker
                ax.scatter(x_vals, y_vals, c=colors, marker=marker, s=80, 
                        alpha=0.7, edgecolors='black', linewidth=0.5,
                        label=f'Episode Length={length_value}')
        
        # Set plot labels and title
        ax.set_xlabel('State-Action Pair Index (n_state × n_actions)', fontsize=12)
        ax.set_ylabel('Visit Count', fontsize=12)
        ax.set_title('State-Action Pair Visit Frequencies', fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set y-axis minimum to 0
        if all_y_values:
            ax.set_ylim(bottom=-0.5, top=max(all_y_values) * 1.1)
        
        eps_legend_patches = [patches.Patch(color=eps_color_map[eps], label=f'ε={eps}') 
                            for eps in epsilon_values]
        
        # Create combined legend
        handles, labels = ax.get_legend_handles_labels()
        all_handles = eps_legend_patches + handles
        all_labels = [f'ε={eps}' for eps in epsilon_values] + labels
        
        ax.legend(all_handles, all_labels, loc='upper right', fontsize=9, 
                framealpha=0.9, bbox_to_anchor=(1.35, 1))
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.85, 1])
       
        # Save figure
        if self.result_dir is not None:
            plt.savefig(os.path.join(self.result_dir, 'state_action_visit_scatter.png'), 
                        dpi=300, bbox_inches='tight')
        
        plt.show()
                

    def visualize_grid_world(self, env, policy, value_function=None, title="", ax=None):
        """
        Visualization logic implemented with specific coordinate mapping
        
        Parameters:
        -----------
        env : object
            Environment object with attributes: n_states, size, target_states, 
            forbidden_states, actions
        policy : np.ndarray
            Policy array, can be 1D (action indices) or 2D (action probabilities)
        value_function : np.ndarray, optional
            Value function array for each state
        title : str
            Plot title
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure
        save_name : str, optional
            Filename to save the visualization. If provided and result_dir is set,
            saves the plot to result_dir/save_name
        
        Returns:
        --------
        fig : matplotlib.figure.Figure or None
            Figure object if ax is None, otherwise None
        ax : matplotlib.axes.Axes
            Axes object containing the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
        else:
            fig = ax.figure
        
        for state in range(env.n_states):
            row = state // env.size
            col = state % env.size
            x_center = col + 0.5
            y_center = env.size - row - 0.5  # Map to drawing system (row 0 at top)
            
            # 1. Determine Color Fill
            if state in env.target_states:
                facecolor = self.colors['target']
            elif state in env.forbidden_states:
                facecolor = self.colors['forbidden']
            else:
                facecolor = self.colors['normal']
            
            # 2. Draw Cell
            rect = patches.Rectangle((col, env.size - row - 1), 1, 1,
                                   facecolor=facecolor, edgecolor='black', alpha=0.7)
            ax.add_patch(rect)
            
            # 3. State Number
            ax.text(x_center, y_center, str(state), ha='center', va='center',
                    fontsize=10, color='black', weight='bold')
            
            # 4. Value Function Display
            if value_function is not None:
                ax.text(x_center, y_center + 0.3, f'{value_function[state]:.2f}',
                        ha='center', va='center', fontsize=8, color='blue')
            
            # 5. Policy Arrow Logic
            # Handle both index-based and probability-based policies
            if policy.ndim > 1:
                action_probs = policy[state]
            else:
                action_probs = np.eye(env.n_actions)[policy[state]]
                
            max_prob = np.max(action_probs)
            # Find action name for the highest probability
            max_action_indices = [i for i, prob in enumerate(action_probs) if prob == max_prob]
            
            if max_action_indices:
                # Use the first best action found
                action_name = env.actions[max_action_indices[0]]
                action_symbol = self.arrows.get(action_name, '?')
                ax.text(x_center, y_center - 0.2, action_symbol,
                        ha='center', va='center', fontsize=16, color='black', weight='bold')
    
        # Final formatting
        ax.set_xlim(0, env.size)
        ax.set_ylim(0, env.size)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save figure if result_dir and save_name are provided
        if self.result_dir is not None:
            save_path = os.path.join(self.result_dir, "mc_policy_comparison")
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax