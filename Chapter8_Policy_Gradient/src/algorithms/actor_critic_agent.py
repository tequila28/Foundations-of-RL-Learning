import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List, Tuple, Dict

from Chapter8_Policy_Gradient.src.algorithms.feature_extractor import FeatureExtractor
from Chapter8_Policy_Gradient.src.algorithms.policy_network import PolicyNetwork, ValueNetwork


class ActorCriticAgent:
    """Actor-Critic algorithm implementation (with feature extractor)"""

    def __init__(self,
                 env,
                 feature_type: str = "one_hot",  # Feature type parameter
                 hidden_size: int = 128,
                 actor_lr: float = 0.001,
                 critic_lr: float = 0.001,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 device: str = 'cpu'):
        """
        Initialize the Actor-Critic algorithm
        
        Args:
            env: The environment instance
            feature_type: Type of feature extraction (e.g., "one_hot")
            hidden_size: Number of neurons in hidden layers
            actor_lr: Learning rate for the actor network
            critic_lr: Learning rate for the critic network
            gamma: Discount factor for future rewards
            gae_lambda: Lambda parameter for Generalized Advantage Estimation
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.feature_type = feature_type
        
        # Feature extractor to convert state to feature representation
        self.feature_extractor = FeatureExtractor(env, feature_type=feature_type)
        self.feature_dim = self.feature_extractor.get_feature_dim()
        
        # Action dimension (number of possible actions in the environment)
        self.action_dim = env.n_actions

        # Actor network (policy network) outputs action probabilities
        self.actor = PolicyNetwork(
            state_dim=self.feature_dim,
            action_dim=self.action_dim,
            hidden_size=hidden_size
        ).to(device)

        # Critic network (value function network) estimates state values
        self.critic = ValueNetwork(
            state_dim=self.feature_dim,
            hidden_size=hidden_size
        ).to(device)

        # Optimizers for training the networks
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr
        )

        # Experience buffer to store trajectories
        self.reset_buffer()

    def reset_buffer(self):
        """Reset the experience buffer to store new trajectories"""
        self.states = []  # Original state indices
        self.state_features = []  # Feature representations of states
        self.actions = []  # Actions taken
        self.rewards = []  # Rewards received
        self.log_probs = []  # Log probabilities of taken actions
        self.values = []  # State value estimates

    def get_state_features(self, state: int) -> torch.Tensor:
        """
        Get feature representation of a state
        
        Args:
            state: The state index
            
        Returns:
            Tensor: Feature representation of the state
        """
        features = self.feature_extractor.extract_features(state)
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def select_action(self, state: int) -> int:
        """
        Select an action according to the current policy
        
        Args:
            state: Current state index
            
        Returns:
            int: Selected action index
        """
        # Get state features for network input
        state_features = self.get_state_features(state)
        
        # Get action probabilities from actor network
        action_probs = self.actor(state_features)
        
        # Get state value estimate from critic network
        with torch.no_grad():
            value = self.critic(state_features)
        
        # Sample an action from the probability distribution
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Save log probability for policy gradient calculation
        log_prob = dist.log_prob(action)
        
        # Store trajectory data in buffer
        self.states.append(state)
        self.state_features.append(state_features)
        self.actions.append(action.item())
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        return action.item()

    def store_reward(self, reward: float):
        """Store a reward in the experience buffer
        
        Args:
            reward: Reward value to store
        """
        self.rewards.append(reward)

    def compute_gae_advantages(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Returns:
            Tuple[Tensor, Tensor]: (advantages, returns)
            - advantages: Advantage estimates for each time step
            - returns: Estimated returns for each time step
        """
        n_steps = len(self.rewards)
        if n_steps < 2:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        
        # Convert rewards to tensor
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        
        # Extract and stack value estimates
        value_tensors = []
        for v in self.values:
            if isinstance(v, torch.Tensor):
                value_tensors.append(v)
            else:
                value_tensors.append(torch.tensor(v, dtype=torch.float32, device=self.device))
        
        values = torch.stack(value_tensors).squeeze(-1)
        
        # Ensure lengths match between rewards and values
        min_len = min(len(rewards), len(values))
        if min_len < 2:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        
        rewards = rewards[:min_len]
        values = values[:min_len]
        
        # Initialize advantage array (length min_len-1, because we need V(t+1))
        advantages = torch.zeros(min_len - 1, device=self.device)
        
        # Compute advantages backward in time
        gae = 0.0
        for t in reversed(range(min_len - 1)):
            # Compute TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            # Update GAE: A_t = δ_t + γλA_{t+1}
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
        
        # Compute returns: R_t = A_t + V(s_t)
        returns = advantages + values[:-1]
        
        # Standardize advantages to reduce variance
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def update_networks(self) -> Tuple[float, float]:
        """
        Update Actor and Critic network parameters using collected trajectories
        
        Returns:
            Tuple[float, float]: (actor_loss, critic_loss)
        """
        if len(self.rewards) < 2:
            return 0.0, 0.0
        
        # Compute GAE advantages and returns
        advantages, returns = self.compute_gae_advantages()
        
        if len(advantages) == 0:
            return 0.0, 0.0
        
        n_usable = len(advantages)  # Number of usable steps for updates
        
        # Recompute log probabilities and values to ensure gradients are correct
        new_log_probs = []
        new_values = []
        
        for i in range(n_usable):
            state_feat = self.state_features[i]
            action = torch.tensor([self.actions[i]], device=self.device)
            
            # Recompute action probabilities for gradient calculation
            action_probs = self.actor(state_feat)
            dist = Categorical(action_probs)
            log_prob = dist.log_prob(action)
            new_log_probs.append(log_prob)
            
            # Recompute state value for gradient calculation
            value = self.critic(state_feat)
            new_values.append(value)
        
        # Stack tensors for batch processing
        log_probs = torch.stack(new_log_probs).squeeze(-1)
        values = torch.stack(new_values).squeeze(-1)
        
        # Actor loss: maximize expected return (negative for gradient ascent)
        advantages = advantages.detach()  # Detach to prevent critic gradient flow
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic loss: minimize value function error
        returns = returns.detach()  # Detach to prevent actor gradient flow
        critic_loss = nn.MSELoss()(values, returns)
        
        # Update Actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        # Update Critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        # Clear storage for next episode
        self.reset_buffer()
        
        return actor_loss.item(), critic_loss.item()

    def train(self, n_episodes: int, max_steps: int = 500) -> Dict[str, List]:
        """
        Train the Actor-Critic algorithm for multiple episodes
        
        Args:
            n_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            
        Returns:
            Dict[str, List]: Training statistics
        """
        episode_rewards = []
        episode_lengths = []
        actor_losses = []
        critic_losses = []
        
        for episode in range(n_episodes):
            # Reset buffer for new episode
            self.reset_buffer()
            
            # Start from a random state
            state = np.random.randint(0, self.env.n_states)
            total_reward = 0
            
            # Run episode
            for step in range(max_steps):
                # Select action using current policy
                action_idx = self.select_action(state)
                action = self.env.actions[action_idx]
                
                # Execute action in environment
                next_state, hit_wall = self.env.transition_logic(state, action)
                reward = self.env.get_reward(next_state, hit_wall)
                
                # Store reward
                self.store_reward(reward)
                
                # Update statistics
                total_reward += reward
                
                # Transition to next state
                state = next_state
            
            # Update networks after episode
            actor_loss, critic_loss = self.update_networks()
            
            # Record statistics
            episode_rewards.append(total_reward)
            episode_lengths.append(max_steps)  # Always max_steps (full episode)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            
            # Print progress periodically
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1}/{n_episodes}, "
                      f"Average Reward: {avg_reward:.2f}, "
                      f"Actor Loss: {actor_loss:.4f}, "
                      f"Critic Loss: {critic_loss:.4f}")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'actor_losses': actor_losses,
            'critic_losses': critic_losses
        }

    def get_policy(self) -> np.ndarray:
        """
        Get the current policy for all states
        
        Returns:
            np.ndarray: Policy matrix of shape (n_states, n_actions)
        """
        policy = np.zeros((self.env.n_states, self.env.n_actions))
        
        for state in range(self.env.n_states):
            state_features = self.get_state_features(state)
            with torch.no_grad():
                probs = self.actor(state_features)
                policy[state] = probs.cpu().numpy()
        
        return policy

    def get_value_estimate(self, state: int) -> float:
        """
        Get value estimate of a state
        
        Args:
            state: State index
            
        Returns:
            float: Estimated value of the state
        """
        state_features = self.get_state_features(state)
        
        with torch.no_grad():
            value = self.critic(state_features)
        
        return value.item()

    def get_value_function(self) -> np.ndarray:
        """
        Get the complete state value function for all states
        
        Returns:
            np.ndarray: Value estimates for all states
        """
        values = np.zeros(self.env.n_states)
        
        for state in range(self.env.n_states):
            state_features = self.get_state_features(state)
            with torch.no_grad():
                value = self.critic(state_features)
                values[state] = value.item()
        
        return values
    
    def get_action_probabilities(self, state: int) -> np.ndarray:
        """
        Get action probabilities for a given state
        
        Args:
            state: State index
            
        Returns:
            np.ndarray: Action probabilities
        """
        state_features = self.get_state_features(state)
        
        with torch.no_grad():
            probs = self.actor(state_features)
        
        return probs.cpu().numpy()
    
    def save_model(self, path: str):
        """Save the model parameters to a file
        
        Args:
            path: File path to save the model
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'feature_type': self.feature_type
        }, path)
    
    def load_model(self, path: str):
        """Load the model parameters from a file
        
        Args:
            path: File path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.feature_type = checkpoint.get('feature_type', 'one_hot')
        
        # Reinitialize feature extractor with saved type
        self.feature_extractor = FeatureExtractor(
            self.env, 
            feature_type=self.feature_type
        )
        
        # Update feature dimension
        self.feature_dim = self.feature_extractor.get_feature_dim()