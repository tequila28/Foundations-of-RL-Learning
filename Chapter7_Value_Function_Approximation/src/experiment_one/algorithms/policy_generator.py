import numpy as np
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")


class EpisodeGenerator:
    """Episode Generator for generating episodes"""
    def __init__(self, env, seed: int = None):
        """
        Initialize the Episode Generator
        
        Parameters:
        -----------
        env : GridWorld
            Grid world environment
        seed : int, optional
            Random seed for policy initialization
        """
        self.env = env
        
        if seed is not None:
            np.random.seed(seed)
        
        self.policy = np.random.rand(self.env.n_states, self.env.n_actions)
        
        row_sums = self.policy.sum(axis=1, keepdims=True) + 1e-8
        self.policy = self.policy / row_sums
        
    def generate_episode(self, max_steps: int = 50) -> List[Tuple[int, int, float]]:
        """
        Generate an episode data
        
        Parameters:
        -----------
        max_steps : int
            Maximum number of steps in the episode
            
        Returns:
        --------
        List[Tuple[int, int, float]]
            List containing (state, action index, reward) tuples
        """
        # Randomly select initial state
        valid_states = [s for s in range(self.env.n_states)]
        
        if valid_states:
            state = np.random.choice(valid_states)
        else:
            state = np.random.randint(0, self.env.n_states)
        
        # Select action according to policy
        action_idx = np.random.choice(range(self.env.n_actions), p=self.policy[state])
        episode = []
        
        for _ in range(max_steps):
            # Get action name
            action = self.env.actions[action_idx]
            
            # Get next state and reward
            next_state, hit_wall = self.env.transition_logic(state, action)
            reward = self.env.get_reward(next_state, hit_wall)
            
            episode.append((state, action_idx, reward))
            state = next_state
            
            # Select next action according to policy
            action_idx = np.random.choice(range(self.env.n_actions), p=self.policy[state])
            
        return episode