"""
SARSA algorithm with Linear Function Approximation.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from tqdm import tqdm


class SARSAFA:
    """SARSA algorithm with Linear Function Approximation."""
    
    def __init__(self, 
                 env,
                 state_feature_extractor,
                 learning_rate: float = 0.1,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 gamma: float = 0.9,
                 initial_theta: Optional[np.ndarray] = None,
                 track_history: bool = False):
        """
        Initialize SARSA algorithm with function approximation.
        
        Args:
            env: GridWorld environment instance
            state_feature_extractor: StateFeatureExtractor instance
            learning_rate: α, parameter update step size
            epsilon: ε, exploration rate for ε-greedy
            epsilon_decay: Decay factor for ε after each episode
            epsilon_min: Minimum exploration rate
            gamma: γ, discount factor
            initial_theta: Initial parameter vector for all actions (if None, random initialization)
            track_history: Whether to track training history
        """
        self.env = env
        self.feat_extractor = state_feature_extractor
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.track_history = track_history
        
        # State feature dimension
        self.state_feature_dim = state_feature_extractor.get_feature_dimension()
        self.n_actions = env.n_actions
        
        # Initialize parameter matrix θ (n_actions x state_feature_dim)
        # Each row corresponds to parameters for one action
        if initial_theta is not None:
            if initial_theta.shape == (self.n_actions, self.state_feature_dim):
                self.theta = initial_theta.copy()
            elif initial_theta.shape == (self.n_actions * self.state_feature_dim,):
                # Reshape if provided as flattened vector
                self.theta = initial_theta.reshape(self.n_actions, self.state_feature_dim)
            else:
                raise ValueError(f"initial_theta must have shape ({self.n_actions}, {self.state_feature_dim}) "
                               f"or ({self.n_actions * self.state_feature_dim},)")
        else:
            # Random initialization with small values
            self.theta = np.random.randn(self.n_actions, self.state_feature_dim) * 0.1
        
        # Initialize training history
        if track_history:
            self.history = {
                'episode_rewards': [],
                'episode_steps': [],
                'td_errors': [],
                'theta_norms': [] 
            }
    
    def get_state_features(self, state: int) -> np.ndarray:
        """
        Extract features for a state.
        
        Args:
            state: State index
            
        Returns:
            State feature vector
        """
        return self.feat_extractor.extract_state_features(state)
    
    def get_q_value(self, state: int, action: int) -> float:
        """
        Compute Q(s,a) = θ_a^T φ(s)
        
        Args:
            state: State index
            action: Action index
            
        Returns:
            Q-value estimate
        """
        features = self.get_state_features(state)
        return np.dot(self.theta[action], features)
    
    def get_q_values(self, state: int) -> np.ndarray:
        """
        Compute Q-values for all actions in a state.
        
        Args:
            state: State index
            
        Returns:
            Array of Q-values for all actions
        """
        features = self.get_state_features(state)
        q_values = np.zeros(self.n_actions)
        
        for action in range(self.n_actions):
            q_values[action] = np.dot(self.theta[action], features)
        
        return q_values
    
    def epsilon_greedy_action(self, state: int, greedy: bool = False) -> int:
        """
        ε-greedy action selection.
        
        Args:
            state: Current state index
            greedy: If True, always use greedy action
            
        Returns:
            Selected action index
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.env.n_actions)
        else:
            # Exploit: greedy action (break ties randomly)
            q_values = self.get_q_values(state)
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
    
    def sarsa_update(self,
                     state: int,
                     action: int,
                     reward: float,
                     next_state: int,
                     next_action: int) -> float:
        """
        SARSA update rule with function approximation:
        θ_a ← θ_a + α * δ * φ(s)
        where δ = r + γ * Q(s',a') - Q(s,a)

        """
        # Get state features
        features = self.get_state_features(state)
        
        # Current Q-value
        current_q = self.get_q_value(state, action)
        
        # Next Q-value
        next_q = self.get_q_value(next_state, next_action)
        
        # TD target and error
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        
        # Update parameters for the specific action: θ_a ← θ_a + α * δ * φ(s)
        self.theta[action] += self.lr * td_error * features
        
        return td_error
    
    def train_episode(self, max_steps: int = 100) -> Tuple[float, int, Optional[List[float]]]:
        """
        Train for one episode using SARSA with function approximation.
        
        Args:
            max_steps: Maximum steps per episode
            
        Returns:
            Tuple of (total reward, number of steps, list of TD errors)
        """
        # Start from a random state
        state = np.random.randint(self.env.size * self.env.size)
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
            
            # Move to next state-action pair
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
            # 修改：记录每个动作的参数范数
            theta_norms = [np.linalg.norm(self.theta[a]) for a in range(self.n_actions)]
            self.history['theta_norms'].append(theta_norms)
        
        return total_reward, steps, avg_td_error
    
    def train(self, 
              num_episodes: int = 1000,
              max_steps: int = 100,
              verbose: bool = False) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Train the SARSA agent with function approximation.
        
        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Whether to print training progress
            
        Returns:
            Tuple of (trained parameters, training history)
        """
        for episode in tqdm(range(num_episodes), desc="Training SARSA with FA"):
            total_reward, steps, avg_td_error = self.train_episode(max_steps)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_theta_norm = np.mean([np.linalg.norm(self.theta[a]) for a in range(self.n_actions)])
                print(f"Episode {episode+1}/{num_episodes}: "
                      f"Reward={total_reward:.2f}, "
                      f"Steps={steps}, "
                      f"TD Error={avg_td_error:.4f}, "
                      f"ε={self.epsilon:.4f}, "
                      f"Avg ‖θ‖={avg_theta_norm:.4f}")
        
        return self.theta, self.history if self.track_history else None
    
    def get_greedy_policy(self) -> np.ndarray:
        """
        Extract greedy policy from learned Q-function.
        
        Returns:
            Greedy policy matrix (n_states, n_actions)
        """
        n_states = self.env.size * self.env.size
        policy = np.zeros((n_states, self.env.n_actions))
        
        for s in range(n_states):
            # Get Q-values for all actions
            q_values = self.get_q_values(s)
            
            # Greedy action (break ties randomly)
            best_action = np.random.choice(np.where(q_values == np.max(q_values))[0])
            policy[s, best_action] = 1.0
        
        return policy
    
    
    def get_value_function(self) -> np.ndarray:
        """
        Get the value function V(s) = max_a Q(s,a) for all states.
        
        Returns:
            Value function array
        """
        n_states = self.env.size * self.env.size
        V = np.zeros(n_states)
        
        for s in range(n_states):
            q_values = self.get_q_values(s)
            V[s] = np.max(q_values)
        
        return V
    
    def get_best_action(self, state: int) -> int:
        """
        Get greedy action for a state (break ties randomly).
        
        Args:
            state: State index
            
        Returns:
            Best action index
        """
        q_values = self.get_q_values(state)
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)
    
    def get_parameters_for_action(self, action: int) -> np.ndarray:
        """
        Get parameters for a specific action.
        
        Args:
            action: Action index
            
        Returns:
            Parameter vector for the specified action
        """
        if 0 <= action < self.n_actions:
            return self.theta[action].copy()
        else:
            raise ValueError(f"Action index {action} out of range [0, {self.n_actions-1}]")
    
    
    def get_all_parameters(self) -> np.ndarray:
        """
        Get all parameters as a 2D array.
        
        Returns:
            Parameter matrix (n_actions x state_feature_dim)
        """
        return self.theta.copy()

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
                'td_errors': [],
                'theta_norms': []
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
            current_q = self.get_q_value(state, action)
            next_q = self.get_q_value(next_state, next_action)
            td_target = reward + self.gamma * next_q
            td_error = abs(td_target - current_q)
            total_td_error += td_error
        
        return total_td_error / len(episode)