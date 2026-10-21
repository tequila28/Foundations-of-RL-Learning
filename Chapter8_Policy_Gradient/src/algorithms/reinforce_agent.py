import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List, Dict

from Chapter8_Policy_Gradient.src.algorithms.feature_extractor import FeatureExtractor
from Chapter8_Policy_Gradient.src.algorithms.policy_network import PolicyNetwork


class REINFORCEAgent:
    """REINFORCE Algorithm Implementation (with feature extractor)"""

    def __init__(self,
                 env,
                 feature_type: str = "one_hot",  # Added: feature type parameter
                 hidden_size: int = 128,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 device: str = 'cpu'):
        """
        Initialize REINFORCE algorithm
        
        Args:
            env: Environment
            feature_type: Type of feature extraction ("one_hot" or "polynomial")
            hidden_size: Hidden layer size of policy network
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            device: Computation device ('cpu' or 'cuda')
        """
        self.env = env
        self.gamma = gamma
        self.device = device
        self.feature_type = feature_type
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(env, feature_type=feature_type)
        
        # Modified: Set feature order based on feature type
        if feature_type == "polynomial":        
            feature_order = 3  # Default polynomial feature order
            self.feature_order = feature_order
        else:
            self.feature_order = None  # one-hot features don't need order
        
        # Modified: Get feature dimension from feature extractor
        self.feature_dim = self.feature_extractor.get_feature_dim()
        
        # State and action dimensions
        self.action_dim = env.n_actions

        # Policy network
        self.policy_network = PolicyNetwork(
            state_dim=self.feature_dim,
            action_dim=self.action_dim,
            hidden_size=hidden_size
        ).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate
        )

        # Experience buffer
        self.reset_buffer()

    def reset_buffer(self):
        """Reset experience buffer"""
        self.states = []
        self.state_features = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def get_state_features(self, state: int) -> torch.Tensor:
        """Get feature representation of state"""
        # Modified: Use unified feature extraction method
        features = self.feature_extractor.extract_features(state)
        features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        return features_tensor

    def select_action(self, state: int) -> int:
        """Select action according to current policy"""
        state_features = self.get_state_features(state)
        action_probs = self.policy_network(state_features)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        self.states.append(state)
        self.state_features.append(state_features)
        self.actions.append(action.item())
        self.log_probs.append(log_prob)
        
        return action.item()

    def store_reward(self, reward: float):
        """Store reward"""
        self.rewards.append(reward)

    def compute_returns(self) -> torch.Tensor:
        """Compute discounted returns"""
        if not self.rewards:
            return torch.tensor([], device=self.device)
        
        returns = []
        R = 0
        
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        return returns

    def update_policy(self) -> float:
        """
        Update policy network parameters
        
        Returns:
            Policy loss value
        """
        if len(self.states) == 0:
            return 0.0
        
        # Calculate discounted returns
        returns = self.compute_returns()
        
        # Normalize returns (helps stabilize training)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate loss
        log_probs = torch.stack(self.log_probs)
        policy_loss = -(log_probs * returns).sum() / len(log_probs)
        
        # Backpropagation
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        # Reset buffer
        self.reset_buffer()
        
        return policy_loss.item()

    def train(self, n_episodes: int, max_steps: int = 500) -> Dict[str, List]:
        """Train REINFORCE algorithm
        
        Args:
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary containing training statistics
        """
        episode_rewards = []
        episode_lengths = []
        policy_entropies = []
        
        for episode in range(n_episodes):
            # Random initial state
            state = np.random.randint(0, self.env.n_states)
            total_reward = 0
            steps = 0
            
            # Collect trajectory
            for step in range(max_steps):
                # Select action
                action_idx = self.select_action(state)
                action = self.env.actions[action_idx]
                
                # Execute action
                next_state, hit_wall = self.env.transition_logic(state, action)
                reward = self.env.get_reward(next_state, hit_wall)
                
                # Store reward
                self.store_reward(reward)
                
                # Update statistics
                total_reward += reward
                state = next_state
                steps = step + 1
                
            
            # Update policy
            loss = self.update_policy()
            
            # Calculate policy entropy
            with torch.no_grad():
                state_features = self.get_state_features(0)
                probs = self.policy_network(state_features)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
            
            # Record statistics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            policy_entropies.append(entropy)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1}/{n_episodes}, "
                      f"Average Reward: {avg_reward:.2f}, "
                      f"Steps: {steps}, "
                      f"Loss: {loss:.4f}")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'policy_entropies': policy_entropies
        }
    
    def get_policy(self) -> np.ndarray:
        """Get current policy
        
        Returns:
            Policy matrix of shape (n_states, n_actions)
        """
        policy = np.zeros((self.env.n_states, self.env.n_actions))
        
        for state in range(self.env.n_states):
            state_features = self.get_state_features(state)
            with torch.no_grad():
                probs = self.policy_network(state_features)
                policy[state] = probs.cpu().numpy()
        
        return policy
    
    def get_action_probabilities(self, state: int) -> np.ndarray:
        """Get action probabilities for a given state
        
        Args:
            state: Current state index
            
        Returns:
            Action probability distribution
        """
        state_features = self.get_state_features(state)
        with torch.no_grad():
            probs = self.policy_network(state_features)
        return probs.cpu().numpy()
    
    def save_model(self, path: str):
        """Save model
        
        Args:
            path: File path to save model
        """
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_type': self.feature_type,
            'feature_order': self.feature_order
        }, path)
    
    def load_model(self, path: str):
        """Load model
        
        Args:
            path: File path to load model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.feature_type = checkpoint.get('feature_type', 'one_hot')
        self.feature_order = checkpoint.get('feature_order')