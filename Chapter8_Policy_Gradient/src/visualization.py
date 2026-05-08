import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Optional, Union, Any
import os
import sys

# Configure environment path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(os.path.dirname(parent_dir))

from Chapter1_Basic_Concepts.src.environment_model import GridWorld


class GridWorldVisualizer:
    """Visualization class for GridWorld policy gradient experiments."""

    def __init__(self, results_dir: str = "results"):
        """
        Initialize the visualizer.

        Args:
            results_dir: Directory to save visualization results
        """
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Colors for different cell types
        self.colors = {
            'normal': 'white',
            'forbidden': 'orange',
            'target': 'lightblue',
        }
        
        # Arrow symbols for actions
        self.arrows = {
            'up': '↑',
            'right': '→',
            'down': '↓',
            'left': '←',
            'stay': '○'
        }
        
        # Colors for different actions
        self.action_colors = {
            'up': 'red',
            'right': 'blue',
            'down': 'green',
            'left': 'orange',
            'stay': 'purple'
        }
        
        # Color mapping for different algorithms
        self.algorithm_colors = {
            'reinforce': '#2E86AB',
            'actor_critic': '#A23B72',
            'comparison': '#F18F01'
        }

    def visualize_grid_world(self, env, policy, value_function=None, title="", ax=None):
        """
        Visualize the GridWorld environment
        
        Parameters:
        -----------
        env : GridWorld
            Environment object containing attributes: n_states, size, target_states, forbidden_states, actions
        policy : np.ndarray
            Policy array, can be 1D (action indices) or 2D (action probabilities)
        value_function : np.ndarray, optional
            Value function array for each state
        title : str
            Chart title
        ax : matplotlib.axes.Axes, optional
            Axis to draw on. If None, create a new figure
            
        Returns:
        --------
        fig : matplotlib.figure.Figure or None
            Returns the figure object if ax is None, otherwise returns None
        ax : matplotlib.axes.Axes
            Axis object containing the plot
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
            
            # 1. Determine cell fill color
            if state in env.target_states:
                facecolor = self.colors['target']
            elif state in env.forbidden_states:
                facecolor = self.colors['forbidden']
            else:
                facecolor = self.colors['normal']
            
            # 2. Draw the cell
            rect = patches.Rectangle((col, env.size - row - 1), 1, 1,
                                   facecolor=facecolor, edgecolor='black', 
                                   alpha=0.7, linewidth=2)
            ax.add_patch(rect)
            
            # 3. State number
            ax.text(x_center, y_center, str(state), 
                   ha='center', va='center',
                   fontsize=10, color='black', weight='bold')
            
            # 4. Value function display
            if value_function is not None and len(value_function) > state:
                ax.text(x_center, y_center + 0.3, f'{value_function[state]:.2f}',
                       ha='center', va='center', 
                       fontsize=8, color='blue', weight='bold')
            
            # 5. Policy arrow logic
            # Handle both index-based and probability-based policies
            if policy is not None:
                if policy.ndim > 1:
                    action_probs = policy[state]
                else:
                    action_probs = np.zeros(len(env.actions))
                    if state < len(policy):
                        action_probs[policy[state]] = 1.0
                
                # Find the action with highest probability
                max_prob = np.max(action_probs)
                max_action_indices = [i for i, prob in enumerate(action_probs) if prob == max_prob]
                
                if max_action_indices and max_prob > 0:
                    # Use the first found best action
                    action_name = env.actions[max_action_indices[0]]
                    action_symbol = self.arrows.get(action_name, '?')
                    action_color = self.action_colors.get(action_name, 'black')
                    
                    # Draw action symbol
                    ax.text(x_center, y_center - 0.2, action_symbol,
                           ha='center', va='center', 
                           fontsize=20, color=action_color, weight='bold')
                    
                    # Display action probability (if not deterministic policy)
                    if max_prob < 0.99 and action_probs.ndim > 0:
                        ax.text(x_center, y_center - 0.4, f'{max_prob:.2f}',
                               ha='center', va='center',
                               fontsize=8, color=action_color)
        
        # Final formatting
        ax.set_xlim(0, env.size)
        ax.set_ylim(0, env.size)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend
        legend_patches = [
            patches.Patch(color=self.colors['normal'], alpha=0.7, label='Normal State'),
            patches.Patch(color=self.colors['forbidden'], alpha=0.7, label='Forbidden State'),
            patches.Patch(color=self.colors['target'], alpha=0.7, label='Target State'),
        ]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=9)
        
        return fig, ax


    def plot_training_curves(self, episode_rewards: List[float],
                            episode_lengths: List[int],
                            policy_entropies: List[float],
                            title: str = "Training Curves",
                            save: bool = True) -> plt.Figure:
        """
        Plot training curves for REINFORCE algorithm.

        Args:
            episode_rewards: List of episode rewards
            episode_lengths: List of episode lengths
            policy_entropies: List of policy entropies
            title: Figure title
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        episodes = range(1, len(episode_rewards) + 1)
        window_size = min(20, max(1, len(episode_rewards) // 10))
        
        metrics = [
            (episode_rewards, 'Total Reward', 'blue', 'red', 'Episode Rewards'),
            (policy_entropies, 'Policy Entropy', 'purple', 'brown', 'Policy Entropy')
        ]
        
        for idx, (data, label, raw_color, smooth_color, sub_title) in enumerate(metrics):
            ax = axes[idx]
            
            if data and len(data) > 0:
                
                # Plot smoothed data
                if len(data) >= window_size:
                    data_smooth = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
                    episodes_smooth = episodes[window_size - 1:]
                    ax.plot(episodes_smooth, data_smooth, linewidth=2, color=smooth_color, 
                           label=f'Smoothed (window={window_size})')
                
                ax.set_xlabel('Episode', fontsize=10)
                ax.set_ylabel(label, fontsize=10)
                ax.set_title(sub_title, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.results_dir, f"{title.lower().replace(' ', '_')}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_ac_training_curves(self, episode_rewards: List[float],
                               actor_losses: List[float],
                               critic_losses: List[float],
                               title: str = "Actor-Critic Training Curves",
                               save: bool = True) -> plt.Figure:
        """
        Plot training curves for Actor-Critic algorithm.

        Args:
            episode_rewards: List of episode rewards
            actor_losses: List of actor losses
            critic_losses: List of critic losses
            title: Figure title
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        episodes = range(1, len(episode_rewards) + 1)
        window_size = min(20, max(1, len(episode_rewards) // 10))
        
        # Plot rewards
        ax = axes[0]
        ax.plot(episodes, episode_rewards, linewidth=1, color='blue', alpha=0.5, label='Raw')
        
        if len(episode_rewards) >= window_size:
            rewards_smooth = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')
            episodes_smooth = episodes[window_size - 1:]
            ax.plot(episodes_smooth, rewards_smooth, linewidth=2, color='red', 
                   label=f'Smoothed')
        
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Total Reward', fontsize=10)
        ax.set_title('Episode Rewards', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Plot actor loss
        ax = axes[1]
        if actor_losses and len(actor_losses) > 0:
            ax.plot(episodes, actor_losses, linewidth=1, color='green', alpha=0.5, label='Actor Loss')
            if len(actor_losses) >= window_size:
                actor_losses_smooth = np.convolve(actor_losses, np.ones(window_size) / window_size, mode='valid')
                ax.plot(episodes_smooth, actor_losses_smooth, linewidth=2, color='darkgreen', 
                       label=f'Smoothed')
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Actor Loss', fontsize=10)
        ax.set_title('Actor Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Plot critic loss
        ax = axes[2]
        if critic_losses and len(critic_losses) > 0:
            ax.plot(episodes, critic_losses, linewidth=1, color='purple', alpha=0.5, label='Critic Loss')
            if len(critic_losses) >= window_size:
                critic_losses_smooth = np.convolve(critic_losses, np.ones(window_size) / window_size, mode='valid')
                ax.plot(episodes_smooth, critic_losses_smooth, linewidth=2, color='purple', 
                       label=f'Smoothed')
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Critic Loss', fontsize=10)
        ax.set_title('Critic Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.results_dir, f"{title.lower().replace(' ', '_')}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig


    def analyze_policy(self, env: GridWorld, policy: np.ndarray, title: str = "Policy Analysis") -> None:
        """
        Analyze and print policy characteristics.

        Args:
            env: GridWorld environment
            policy: Policy matrix
            title: Analysis title
        """
        print(f"\n{'='*60}")
        print(f"{title.upper()}")
        print('='*60)
        
        # Calculate policy entropy for each state
        entropy_per_state = -np.sum(policy * np.log(policy + 1e-8), axis=1)
        avg_entropy = np.mean(entropy_per_state)
        
        print(f"\nPolicy Statistics:")
        print(f"  Average Policy Entropy: {avg_entropy:.4f}")
        print(f"  Min Entropy: {np.min(entropy_per_state):.4f}")
        print(f"  Max Entropy: {np.max(entropy_per_state):.4f}")
        
        # Analyze target state behavior
        print(f"\nTarget State Behavior:")
        for state in env.target_states:
            row, col = divmod(state, env.size)
            best_action_idx = np.argmax(policy[state])
            best_action = env.actions[best_action_idx]
            best_prob = policy[state, best_action_idx]
            print(f"  State {state} (Row {row}, Col {col}): "
                  f"Best Action = {best_action} (Prob = {best_prob:.4f})")
        
        # Analyze forbidden state behavior
        print(f"\nForbidden State Behavior:")
        for state in env.forbidden_states:
            row, col = divmod(state, env.size)
            best_action_idx = np.argmax(policy[state])
            best_action = env.actions[best_action_idx]
            best_prob = policy[state, best_action_idx]
            print(f"  State {state} (Row {row}, Col {col}): "
                  f"Best Action = {best_action} (Prob = {best_prob:.4f})")
        
        # Calculate action distribution
        avg_action_probs = np.mean(policy, axis=0)
        print(f"\nAverage Action Distribution:")
        for action_idx, action in enumerate(env.actions):
            print(f"  {action}: {avg_action_probs[action_idx]:.4f}")
        
        # Find most deterministic states
        print(f"\nMost Deterministic States (Entropy < 0.1):")
        deterministic_states = [i for i, entropy in enumerate(entropy_per_state) if entropy < 0.1]
        for state in deterministic_states[:5]:  # Show first 5
            row, col = divmod(state, env.size)
            best_action_idx = np.argmax(policy[state])
            best_action = env.actions[best_action_idx]
            best_prob = policy[state, best_action_idx]
            print(f"  State {state} (Row {row}, Col {col}): "
                  f"{best_action} with {best_prob:.4f} probability")


    def plot_algorithm_comparison(self, reinforce_rewards: List[float],
                             ac_rewards: List[float],
                             title: str = "Algorithm Comparison",
                             save: bool = True) -> plt.Figure:
        """
        Plot comparison of REINFORCE and Actor-Critic performance (smoothed rewards only).

        Args:
            reinforce_rewards: REINFORCE episode rewards
            ac_rewards: Actor-Critic episode rewards
            title: Figure title
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Create only one subplot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        episodes = range(1, len(reinforce_rewards) + 1)
        
        # Calculate appropriate window size for smoothing
        window_size = min(50, max(1, len(reinforce_rewards) // 5))
        
        if len(reinforce_rewards) >= window_size:
            # Apply moving average smoothing
            reinforce_smooth = np.convolve(reinforce_rewards, np.ones(window_size) / window_size, mode='valid')
            ac_smooth = np.convolve(ac_rewards, np.ones(window_size) / window_size, mode='valid')
            episodes_smooth = episodes[window_size - 1:]
            
            # Plot smoothed rewards
            ax.plot(episodes_smooth, reinforce_smooth, linewidth=2, 
                color=self.algorithm_colors['reinforce'], 
                label=f'REINFORCE (window={window_size})')
            ax.plot(episodes_smooth, ac_smooth, linewidth=2, 
                color=self.algorithm_colors['actor_critic'], 
                label=f'Actor-Critic (window={window_size})')
        else:
            # If data is too sparse for smoothing, plot raw data
            ax.plot(episodes, reinforce_rewards, linewidth=2, 
                color=self.algorithm_colors['reinforce'], 
                label='REINFORCE')
            ax.plot(episodes, ac_rewards, linewidth=2, 
                color=self.algorithm_colors['actor_critic'], 
                label='Actor-Critic')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
    
        if save:
            filename = os.path.join(self.results_dir, f"{title.lower().replace(' ', '_')}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_policy_comparison(self, env: GridWorld,
                              reinforce_policy: np.ndarray,
                              ac_policy: np.ndarray,
                              title: str = "Policy Comparison",
                              save: bool = True) -> plt.Figure:
        """
        Plot comparison of REINFORCE and Actor-Critic policies.

        Args:
            env: GridWorld environment
            reinforce_policy: REINFORCE policy matrix
            ac_policy: Actor-Critic policy matrix
            title: Figure title
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot REINFORCE policy
        ax = axes[0]
        self.visualize_grid_world(env, reinforce_policy, 
                                 title="REINFORCE Policy", ax=ax)
        
        # Plot Actor-Critic policy
        ax = axes[1]
        self.visualize_grid_world(env, ac_policy, 
                                 title="Actor-Critic Policy", ax=ax)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.results_dir, f"{title.lower().replace(' ', '_')}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_combined_view(self, env: GridWorld, 
                          reinforce_policy: np.ndarray,
                          ac_policy: np.ndarray,
                          reinforce_rewards: List[float],
                          ac_rewards: List[float],
                          title: str = "Algorithm Comparison") -> plt.Figure:
        """
        Create a comprehensive comparison view.

        Args:
            env: GridWorld environment
            reinforce_policy: REINFORCE policy matrix
            ac_policy: Actor-Critic policy matrix
            reinforce_rewards: REINFORCE episode rewards
            ac_rewards: Actor-Critic episode rewards
            title: Figure title
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create subplot grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Top row: Policies
        ax1 = fig.add_subplot(gs[0, 0])
        self.visualize_grid_world(env, reinforce_policy, 
                                 title="REINFORCE Policy", ax=ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self.visualize_grid_world(env, ac_policy, 
                                 title="Actor-Critic Policy", ax=ax2)
        
        # Bottom row: Training curves
        ax3 = fig.add_subplot(gs[1, :])
        
        episodes = range(1, len(reinforce_rewards) + 1)
        
        # Plot both rewards
        ax3.plot(episodes, reinforce_rewards, linewidth=1.5,
                color=self.algorithm_colors['reinforce'], alpha=0.7,
                label='REINFORCE')
        ax3.plot(episodes, ac_rewards, linewidth=1.5,
                color=self.algorithm_colors['actor_critic'], alpha=0.7,
                label='Actor-Critic')
        
        # Add smoothed versions
        window_size = min(50, max(1, len(reinforce_rewards) // 5))
        if len(reinforce_rewards) >= window_size:
            reinforce_smooth = np.convolve(reinforce_rewards, np.ones(window_size) / window_size, mode='valid')
            ac_smooth = np.convolve(ac_rewards, np.ones(window_size) / window_size, mode='valid')
            episodes_smooth = episodes[window_size - 1:]
            
            ax3.plot(episodes_smooth, reinforce_smooth, linewidth=3,
                    color=self.algorithm_colors['reinforce'], 
                    label=f'REINFORCE (smoothed)')
            ax3.plot(episodes_smooth, ac_smooth, linewidth=3,
                    color=self.algorithm_colors['actor_critic'], 
                    label=f'Actor-Critic (smoothed)')
        
        ax3.set_xlabel('Episode', fontsize=12)
        ax3.set_ylabel('Total Reward', fontsize=12)
        ax3.set_title('Training Performance Comparison', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10, loc='best')
        
        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(self.results_dir, f"combined_comparison.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig