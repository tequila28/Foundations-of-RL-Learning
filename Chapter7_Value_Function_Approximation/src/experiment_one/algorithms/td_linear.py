import numpy as np
from feature_extractor import FeatureExtractor

class TDLinear:
    """TD-Linear algorithm with linear function approximation"""
    def __init__(self, env, param: int, alpha: float = 0.001):
        """
        Initialize TD-Linear algorithm
        
        Parameters:
        -----------
        env : GridWorld
            Grid world environment
        param : int
            Polynomial order (1, 2, or 3)
        alpha : float
            Learning rate
        """
        self.env = env
        self.param = param
        self.alpha = alpha
        self.feature_extractor = FeatureExtractor(env)
        
        # Set feature dimension based on polynomial order
        self.dim = {1: 3, 2: 6, 3: 10}[param]
        self.weights = np.random.randn(self.dim) * 0.1
        
    def get_features(self, state: int) -> np.ndarray:
        """
        Get feature vector for a state
        
        Parameters:
        -----------
        state : int
            State index
            
        Returns:
        --------
        np.ndarray
            Feature vector
        """
        return self.feature_extractor.polynomial_features(state, self.param)
    
    def value_estimate(self, state: int) -> float:
        """
        Estimate state value function
        
        Parameters:
        -----------
        state : int
            State index
            
        Returns:
        --------
        float
            Estimated state value
        """
        return np.dot(self.weights, self.get_features(state))
    
    def update(self, state: int, next_state: int, reward: float):
        """
        Perform TD update
        
        Parameters:
        -----------
        state : int
            Current state
        next_state : int
            Next state
        reward : float
            Received reward
        """
        # Calculate current and next state values
        current_value = self.value_estimate(state)
        next_value = self.value_estimate(next_state)
        
        # Calculate TD error
        td_error = reward + self.env.gamma * next_value - current_value
        
        # Get features
        features = self.get_features(state)
        
        # Gradient clipping
        grad_norm = np.linalg.norm(features)
        if grad_norm > 1.0:
            features = features / grad_norm
        
        # Update weights
        self.weights += self.alpha * td_error * features
        
        # Weight clipping
        weight_norm = np.linalg.norm(self.weights)
        if weight_norm > 5.0:
            self.weights = self.weights / weight_norm * 5.0