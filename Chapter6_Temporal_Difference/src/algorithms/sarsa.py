"""
Standard SARSA algorithm implementation.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from tqdm import tqdm
from Chapter1_Basic_Concepts.src.environment_model import GridWorld

class SARSA:
    """Standard SARSA algorithm implementation."""
    
    def __init__(self, 
                 env: GridWorld,
                 learning_rate: float = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 gamma: float = 0.9,
                 track_history: bool = False):
        """
        Initialize SARSA algorithm.
        
        Args:
            env: GridWorld environment instance
            learning_rate: α, Q-value update step size
            epsilon: ε, exploration rate for ε-greedy
            epsilon_decay: Decay factor for ε after each episode
            epsilon_min: Minimum exploration rate
            gamma: γ, discount factor
            track_history: Whether to track training history
        """
        self.env = env
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.track_history = track_history
        
        # Initialize Q-table
        self.Q = np.zeros((env.n_states, env.n_actions))
        
        # Initialize training history
        if track_history:
            self.history = {
                'episode_rewards': [],
                'episode_steps': [],
                'td_errors': []
            }
    
    def epsilon_greedy_action(self, state: int, greedy: bool = False) -> int:
        """
        ε-greedy action selection.
        
        Args:
            state: Current state index
            greedy: If True, always use greedy action
            
        Returns:
            Selected action index
        """
        if greedy or np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.env.n_actions)
        else:
            # Exploit: greedy action (break ties randomly)
            max_q = np.max(self.Q[state])
            best_actions = np.where(self.Q[state] == max_q)[0]
            return np.random.choice(best_actions)
    
    def sarsa_update(self,
                     state: int,
                     action: int,
                     reward: float,
                     next_state: int,
                     next_action: int) -> float:
        """
        Standard SARSA update rule: Q(s,a) ← Q(s,a) + α[r + γ * Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state observed
            next_action: Next action to be taken
            
        Returns:
            TD error
        """
        # Current Q-value
        current_q = self.Q[state, action]
        
        # Next Q-value
        next_q = self.Q[next_state, next_action]
        
        # TD target
        td_target = reward + self.gamma * next_q
        
        # TD error
        td_error = td_target - current_q
        
        # Update Q-value
        self.Q[state, action] = current_q + self.lr * td_error
        
        return td_error
    
    def train_episode(self, max_steps: int = 100) -> Tuple[float, int, Optional[List[float]]]:
        """
        Train for one episode using standard SARSA.
        
        Args:
            max_steps: Maximum steps per episode
            
        Returns:
            Tuple of (total reward, number of steps, list of TD errors)
        """
        state = np.random.randint(self.env.size * self.env.size)  # Start from a random state
        action = self.epsilon_greedy_action(state)
        total_reward = 0
        steps = 0
        episode_td_errors = []
        
        for _ in range(max_steps):
            # Take action
            next_state_dist = self.env.P[state, action]
            next_state = np.random.choice(len(next_state_dist), p=next_state_dist)
            reward = self.env.R[state, action]
            
            # Choose next action
            next_action = self.epsilon_greedy_action(next_state)
            
            # SARSA update
            td_error = self.sarsa_update(state, action, reward, next_state, next_action)
            episode_td_errors.append(td_error)
            
            # Update tracking
            total_reward += reward
            steps += 1
            
            # Move to next state
            state = next_state
            action = next_action
        
        # Calculate average TD error for the episode
        avg_td_error = np.mean(np.abs(episode_td_errors)) if episode_td_errors else 0.0
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update history if tracking is enabled
        if self.track_history:
            self.history['episode_rewards'].append(total_reward)
            self.history['episode_steps'].append(steps)
            self.history['td_errors'].append(avg_td_error)
        
        return total_reward, steps, avg_td_error
    
    def train(self, 
              num_episodes: int = 1000,
              max_steps: int = 100,
              verbose: bool = False) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Train the SARSA agent.
        
        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Whether to print training progress
            
        Returns:
            Tuple of (trained Q-table, training history)
        """
        for episode in tqdm(range(num_episodes), desc="Training SARSA"):
            total_reward, steps, avg_td_error = self.train_episode(max_steps)
            
            if verbose and (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{num_episodes}: "
                      f"Reward={total_reward:.2f}, "
                      f"Steps={steps}, "
                      f"TD Error={avg_td_error:.4f}, "
                      f"ε={self.epsilon:.4f}")
        
        return self.Q, self.history if self.track_history else None
    
    def get_greedy_policy(self) -> np.ndarray:
        """
        Extract greedy policy from learned Q-values.
        
        Returns:
            Greedy policy matrix (n_states, n_actions)
        """
        policy = np.zeros((self.env.n_states, self.env.n_actions))
        
        for s in range(self.env.n_states):
            # Greedy action
            best_action = np.argmax(self.Q[s])
            policy[s, best_action] = 1.0
        
        return policy
    
    def get_epsilon_greedy_policy(self) -> np.ndarray:
        """
        Extract ε-greedy policy from learned Q-values.
        
        Returns:
            ε-greedy policy matrix (n_states, n_actions)
        """
        policy = np.zeros((self.env.n_states, self.env.n_actions))
        
        for s in range(self.env.n_states):
            # Find best action(s)
            max_q = np.max(self.Q[s])
            best_actions = np.where(self.Q[s] == max_q)[0]
            n_best = len(best_actions)
            
            # ε-greedy probabilities
            for a in range(self.env.n_actions):
                if a in best_actions:
                    # Best action gets (1-ε)/n_best + ε/n_actions
                    policy[s, a] = (1.0 - self.epsilon) / n_best + self.epsilon / self.env.n_actions
                else:
                    # Other actions get ε/n_actions
                    policy[s, a] = self.epsilon / self.env.n_actions
        
        return policy
    
    def get_training_history(self) -> Optional[Dict]:
        """
        Get training history.
        
        Returns:
            Training history dictionary or None if not tracking
        """
        return self.history if self.track_history else None
    
    def reset_history(self) -> None:
        """
        Reset training history.
        """
        if self.track_history:
            self.history = {
                'episode_rewards': [],
                'episode_steps': [],
                'td_errors': []
            }
    
    def calculate_episode_td_error(self, episode: List[Tuple[int, int, int, float, int]]) -> float:
        """
        Calculate average absolute TD error for an episode.
        
        Args:
            episode: List of (state, action, reward, next_state, next_action) transitions
            
        Returns:
            Average absolute TD error
        """
        if not episode:
            return 0.0
        
        total_td_error = 0.0
        for state, action, reward, next_state, next_action in episode:
            current_q = self.Q[state, action]
            next_q = self.Q[next_state, next_action]
            td_target = reward + self.gamma * next_q
            td_error = abs(td_target - current_q)
            total_td_error += td_error
        
        return total_td_error / len(episode)