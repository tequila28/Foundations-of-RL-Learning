import numpy as np
from typing import Union

class FeatureExtractor:
    """Feature extractor for state representation"""
    def __init__(self, env):
        """
        Initialize the Feature Extractor
        
        Parameters:
        -----------
        env : GridWorld
            Grid world environment
        """
        self.env = env
        
    def polynomial_features(self, state: int, order: int) -> np.ndarray:
        """
        Generate polynomial features for a state
        
        Parameters:
        -----------
        state : int
            State index
        order : int
            Polynomial order (1, 2, or 3)
            
        Returns:
        --------
        np.ndarray
            Feature vector
        """
        row, col = state // self.env.size, state % self.env.size
        x, y = col, row
        
        # Normalize to [-0.5, 0.5] range
        x_norm = (x - (self.env.size - 1) * 0.5) / (self.env.size - 1)
        y_norm = (y - (self.env.size - 1) * 0.5) / (self.env.size - 1)
        
        if order == 1:
            return np.array([1.0, x_norm, y_norm])
        elif order == 2:
            return np.array([1.0, x_norm, y_norm, 
                           x_norm**2, x_norm*y_norm, y_norm**2])
        elif order == 3:
            return np.array([1.0, x_norm, y_norm, 
                           x_norm**2, x_norm*y_norm, y_norm**2,
                           x_norm**3, x_norm**2*y_norm, 
                           x_norm*y_norm**2, y_norm**3])
        else:
            raise ValueError("Unsupported order. Supported orders: 1, 2, 3")