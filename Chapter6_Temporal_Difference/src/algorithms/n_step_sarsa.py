"""
N-step SARSA algorithm implementation.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from tqdm import tqdm
from Chapter1_Basic_Concepts.src.environment_model import GridWorld


class NStepSARSA:
    """N-step SARSA algorithm implementation."""
    
    def __init__(self, 
                 env: GridWorld,
                 n: int = 3,
                 learning_rate: float = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 gamma: float = 0.9,
                 track_history: bool = False):
        """
        Initialize n-step SARSA algorithm.
        
        Args:
            env: GridWorld environment instance
            n: Number of steps for n-step SARSA
            learning_rate: α, Q-value update step size
            epsilon: ε, exploration rate for ε-greedy
            epsilon_decay: Decay factor for ε after each episode
            epsilon_min: Minimum exploration rate
            gamma: γ, discount factor
            track_history: Whether to track training history
        """
        self.env = env
        self.n = n
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
    
    def train_episode(self, max_steps: int = 100) -> Tuple[float, int, float]:

        state = np.random.randint(self.env.size * self.env.size)
        action = self.epsilon_greedy_action(state)
        states = [state]
        actions = [action]
        rewards = [0.0]
        
        total_reward = 0
        episode_td_errors = []
        T = float('inf')
        t = 0
        
        while True:
            if t < T:
                next_state_dist = self.env.P[states[t], actions[t]]
                next_state = np.random.choice(len(next_state_dist), p=next_state_dist)
                reward = self.env.R[states[t], actions[t]]
                
                states.append(next_state)
                rewards.append(reward)
                total_reward += reward
                
                if t + 1 >= max_steps: 
                    T = t + 1
                else:
                    next_action = self.epsilon_greedy_action(next_state)
                    actions.append(next_action)
            
            tau = t - self.n + 1
            
            if tau >= 0:
                G = 0
                for i in range(tau + 1, min(tau + self.n, T) + 1):
                    G += (self.gamma ** (i - tau - 1)) * rewards[i]
                
                if tau + self.n < T:
                    state_n = states[tau + self.n]
                    action_n = actions[tau + self.n]
                    G += (self.gamma ** self.n) * self.Q[state_n, action_n]
                
                state_tau = states[tau]
                action_tau = actions[tau]
                current_q = self.Q[state_tau, action_tau]
                
                td_error = G - current_q
                self.Q[state_tau, action_tau] += self.lr * td_error
                episode_td_errors.append(td_error)
                
            if tau == T - 1:
                break
                
            t += 1
        
        # Calculate average TD error for the episode
        avg_td_error = np.mean(np.abs(episode_td_errors)) if episode_td_errors else 0.0
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update history if tracking is enabled
        if self.track_history:
            self.history['episode_rewards'].append(total_reward)
            self.history['episode_steps'].append(min(t, max_steps))
            self.history['td_errors'].append(avg_td_error)
        
        return total_reward, min(t, max_steps), avg_td_error
    
    def train(self, 
              num_episodes: int = 1000,
              max_steps: int = 100,
              verbose: bool = False) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Train the n-step SARSA agent.
        
        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Whether to print training progress
            
        Returns:
            Tuple of (trained Q-table, training history)
        """
        for episode in tqdm(range(num_episodes), desc="Training N-step SARSA"):
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
    
    def calculate_episode_td_error(self, episode_transitions: List[Tuple[int, int, float, int, int]]) -> float:
        """
        Calculate average absolute TD error for an episode.
        
        Args:
            episode_transitions: List of (tau, state_tau, action_tau, reward, state_n, action_n) transitions
            
        Returns:
            Average absolute TD error
        """
        if not episode_transitions:
            return 0.0
        
        total_td_error = 0.0
        for transition in episode_transitions:
            if len(transition) == 6:
                tau, state_tau, action_tau, reward_sum, state_n, action_n = transition
                # Simplified calculation - in practice, this would need the full trajectory
                current_q = self.Q[state_tau, action_tau]
                next_q = self.Q[state_n, action_n] if state_n not in self.env.target_states else 0
                # This is a simplified version; actual n-step TD error calculation is more complex
                td_error = abs(current_q - next_q)
                total_td_error += td_error
        
        return total_td_error / len(episode_transitions)