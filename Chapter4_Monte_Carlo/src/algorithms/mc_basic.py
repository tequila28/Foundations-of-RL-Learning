import numpy as np
from typing import List, Tuple, Dict
import random

class MCBasic:
    """
    Basic Monte Carlo Control Algorithm
    Starts from every state-action pair and generates episode of length n
    No ε-greedy exploration needed
    """
    
    def __init__(self, env, episode_length: int = 10, gamma: float = 0.9):
        """
        Initialize MC basic agent
        
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
        
    def generate_episode_from_state_action(self, start_state: int, start_action: int) -> List[Tuple[int, int, float]]:
        """
        Generate episode starting from specific state-action pair
        
        Args:
            start_state: Starting state index
            start_action: Starting action index
            
        Returns:
            episode: List of (state, action, reward) tuples
        """
        episode = []
        state = start_state
        action_idx = start_action
        
        # First step from the specified state-action pair
        next_state, hit_wall = self.env.transition_logic(state, self.env.actions[action_idx])
        reward = self.env.get_reward(next_state, hit_wall)
        episode.append((state, action_idx, reward))
        state = next_state
        
        # Continue for remaining steps
        for _ in range(self.episode_length - 1):
            # After first step, use greedy policy
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
            episode_num: Current episode number (for logging)
        """
        G = 0.0
        visited = set()
        
        # Calculate returns from the end backwards
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            
            # First-visit MC: only update the first time (state, action) is visited
            sa_pair = (state, action)
            if sa_pair not in visited:
                visited.add(sa_pair)
                self.returns_sum[state, action] += G
                self.returns_count[state, action] += 1
                self.Q[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]
    
    def train(self, num_iterations: int = 1000):
        """
        Train the agent by generating episodes from all state-action pairs
        
        Args:
            num_iterations: Number of training iterations
        """
        print(f"Training MC Basic with episode_length={self.episode_length}...")
        
        for iteration in range(num_iterations):
            if iteration % (num_iterations // 5) == 0:
                print(f"Iteration {iteration}/{num_iterations}")
            
            # For each state-action pair, generate an episode
            for state in range(self.n_states):
                    
                for action in range(self.n_actions):
                    # Generate episode starting from (state, action)
                    episode = self.generate_episode_from_state_action(state, action)
                    
                    # Update Q-values
                    self.update(episode, iteration)
    
    
    def get_results(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the learned value function and policy
        
        Returns:
            V: Value function array (n_states,)
            policy: Policy array (n_states,)
        """
        # Value function is max Q-value for each state
        V = np.max(self.Q, axis=1)
        
        # Policy is greedy with respect to Q
        for s in range(self.n_states):
            max_q = np.max(self.Q[s])
            best_actions = np.where(self.Q[s] == max_q)[0]
            self.policy[s] = np.random.choice(best_actions) if len(best_actions) > 0 else 0
        
        return V, self.policy
    
