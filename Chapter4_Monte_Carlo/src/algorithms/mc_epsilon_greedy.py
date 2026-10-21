import numpy as np
from typing import List, Tuple
import random

class MCEpsilonGreedy:
    """
    Monte Carlo Control with ε-greedy policy
    Uses ε-greedy policy for exploration
    """
    
    def __init__(self, env, epsilon: float = 0.1, gamma: float = 0.9, episode_length: int = 10):
        """
        Initialize MC ε-greedy agent
        
        Args:
            env: Environment object
            epsilon: Probability of random action (exploration)
            gamma: Discount factor
            episode_length: Length of each episode
        """
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.episode_length = episode_length
        
        # Initialize Q-table
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.Q = np.zeros((self.n_states, self.n_actions))
        
        # Initialize policy with epsilon-greedy distribution
        self.policy = np.ones((self.n_states, self.n_actions)) * (self.epsilon / self.n_actions)
        
        # Set greedy action probability
        initial_greedy_action = np.random.randint(0, self.n_actions)
        for state in range(self.n_states):
            self.policy[state, initial_greedy_action] += (1 - self.epsilon)
        
        # For storing return statistics
        self.returns_sum = np.zeros((self.n_states, self.n_actions))
        self.returns_count = np.zeros((self.n_states, self.n_actions))
        
    def choose_action(self, state: int) -> int:
        """
        Choose action using ε-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            action: Action index
        """
        return np.random.choice(self.n_actions, p=self.policy[state])
    
    def generate_episode(self) -> List[Tuple[int, int, float]]:
        """
        Generate episode following ε-greedy policy
        
        Returns:
            episode: List of (state, action, reward) tuples
        """
        episode = []

        # Exploring starts: random initial state-action pair
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
        
        for _ in range(self.episode_length - 1):
            # Choose action using ε-greedy policy
            action_idx = self.choose_action(state)
            
            # Take action
            next_state, hit_wall = self.env.transition_logic(state, self.env.actions[action_idx])
            reward = self.env.get_reward(next_state, hit_wall)
            
            # Record transition
            episode.append((state, action_idx, reward))
            
            # Update state
            state = next_state
            
        return episode
    
    def update_policy(self, state: int):
        """
        Update ε-greedy policy for a state based on Q-values
        
        Args:
            state: State to update policy for
        """
        # Find best action(s)
        max_q = np.max(self.Q[state])
        best_actions = np.where(self.Q[state] == max_q)[0]
        n_best = len(best_actions)
        
        # ε-greedy policy: greedy action(s) get (1-ε)/n_best probability
        # all actions get ε/n_actions exploration probability
       
        self.policy[state] = np.ones(self.n_actions) * (self.epsilon / self.n_actions)
        
        greedy_prob = (1.0 - self.epsilon) / n_best
        for a in best_actions:
            self.policy[state, a] += greedy_prob
    

    def update(self, episode: List[Tuple[int, int, float]], episode_num: int):
        """
        Update Q-values and policy using every-visit Monte Carlo
        
        Args:
            episode: List of (state, action, reward) tuples
            episode_num: Current episode number
        """
      
        G = 0.0
        
        # Calculate returns from the end backwards
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward
            
            # Every-visit MC: update when the time (state, action) is visited
            self.returns_sum[state, action] += G
            self.returns_count[state, action] += 1
            self.Q[state, action] = self.returns_sum[state, action] / self.returns_count[state, action]
            self.update_policy(state)
    
    def train(self, num_iterations: int = 1000):
        """
        Train the agent
        
        Args:
            num_iterations: Number of episodes to train
        """
        print(f"Training MC ε-greedy (ε={self.epsilon}) with {num_iterations} episodes...")
        
        for episode_num in range(num_iterations):
            if episode_num % (num_iterations // 5) == 0:
                print(f"Episode {episode_num}/{num_iterations}")
            
            # Generate episode
            episode = self.generate_episode()
            
            # Update
            self.update(episode, episode_num)
    
    def get_results(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the learned value function and deterministic policy
        
        Returns:
            V: Value function array (n_states,)
            policy: Deterministic greedy policy array (n_states,)
        """
        # Value function is max Q-value
        V = np.max(self.Q, axis=1)
        
        # Deterministic greedy policy
        deterministic_policy = np.zeros(self.n_states, dtype=int)
        for s in range(self.n_states):
            max_q = np.max(self.Q[s])
            best_actions = np.where(self.Q[s] == max_q)[0]
            deterministic_policy[s] = np.random.choice(best_actions) if len(best_actions) > 0 else 0
        
        return V, deterministic_policy
    
    def get_stochastic_policy(self) -> np.ndarray:
        """
        Get the full stochastic policy
        
        Returns:
            policy: Stochastic policy matrix (n_states, n_actions)
        """
        return self.policy.copy()