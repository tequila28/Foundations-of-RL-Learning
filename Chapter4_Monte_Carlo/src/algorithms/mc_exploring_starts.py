import numpy as np
from typing import List, Tuple
import random

class MCExploringStarts:
    """
    Monte Carlo Control with Exploring Starts
    Starts episodes from random state-action pairs
    """
    
    def __init__(self, env, episode_length: int = 10, gamma: float = 0.9):
        """
        Initialize MC Exploring Starts agent
        
        Args:
            env: Environment object
            episode_length: Length of each generated episode
            gamma: Discount factor
        """
        self.env = env
        self.episode_length = episode_length
        self.gamma = gamma
        
        # Initialize Q-table
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.Q = np.zeros((self.n_states, self.n_actions))
        
        # For storing return statistics
        self.returns_sum = np.zeros((self.n_states, self.n_actions))
        self.returns_count = np.zeros((self.n_states, self.n_actions))
        
        # Policy cache
        self.policy = np.zeros(self.n_states, dtype=int)
        
    def generate_episode_with_exploring_starts(self) -> List[Tuple[int, int, float]]:
        """
        Generate episode with exploring starts
        Random starting state-action pair
        
        Returns:
            episode: List of (state, action, reward) tuples
        """
        episode = []
        
        # Exploring starts: random initial state-action pair
        start_state = np.random.choice(self.n_states)
        start_action = np.random.randint(0, self.n_actions)
        
        state = start_state
        action_idx = start_action
        
        # First step from random start
        next_state, hit_wall = self.env.transition_logic(state, self.env.actions[action_idx])
        reward = self.env.get_reward(next_state, hit_wall)
        episode.append((state, action_idx, reward))
        state = next_state
        
        # Continue with greedy policy
        for _ in range(self.episode_length - 1):
            # Use greedy policy after exploring start
            max_q = np.max(self.Q[state])
            best_actions = np.where(self.Q[state] == max_q)[0]
            if len(best_actions) == 0:
                action_idx = np.random.randint(0, self.n_actions)
            else:
                action_idx = np.random.choice(best_actions)
            
            # Take action
            next_state, hit_wall = self.env.transition_logic(state, self.env.actions[action_idx])
            reward = self.env.get_reward(next_state, hit_wall)
            
            # Record transition
            episode.append((state, action_idx, reward))
            
            # Update state
            state = next_state
            
        
        return episode
    
    def update(self, episode: List[Tuple[int, int, float]], episode_num: int):
        """
        Update Q-values using first-visit Monte Carlo
        
        Args:
            episode: List of (state, action, reward) tuples
            episode_num: Current episode number
        """
        G = 0.0
        visited = set()
        
        # Calculate returns from the end backwards
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            
            # First-visit MC
            sa_pair = (state, action)
            if sa_pair not in visited:
                visited.add(sa_pair)
                self.returns_sum[state, action] += G
                self.returns_count[state, action] += 1
                self.Q[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]
    
    def train(self, num_iterations: int = 1000):
        """
        Train the agent with exploring starts
        
        Args:
            num_iterations: Number of episodes to train
        """
        print(f"Training MC Exploring Starts with {num_iterations} episodes...")
        
        for episode_num in range(num_iterations):
            if episode_num % (num_iterations // 5) == 0:
                print(f"Episode {episode_num}/{num_iterations}")
            
            # Generate episode with exploring starts
            episode = self.generate_episode_with_exploring_starts()
            
            # Update Q-values
            self.update(episode, episode_num)
    
    def get_results(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the learned value function and policy
        
        Returns:
            V: Value function array (n_states,)
            policy: Policy array (n_states,)
        """
        # Value function is max Q-value
        V = np.max(self.Q, axis=1)
        
        # Greedy policy
        for s in range(self.n_states):
            max_q = np.max(self.Q[s])
            best_actions = np.where(self.Q[s] == max_q)[0]
            self.policy[s] = np.random.choice(best_actions) if len(best_actions) > 0 else 0
        
        return V, self.policy
    
    def get_visit_counts(self) -> np.ndarray:
        """
        Get state-action visit counts
        
        Returns:
            visit_counts: Matrix of visit counts
        """
        return self.returns_count