import numpy as np
from typing import Union


class FeatureExtractor:
    """Feature extractor supporting One-Hot and Polynomial features"""
    def __init__(self, env, feature_type: str = "one_hot", feature_order: int = 2):
        """
        Initialize the feature extractor
        
        Parameters:
        -----------
        env : GridWorld
            Grid world environment instance
        feature_type : str
            Type of feature: "one_hot" or "polynomial"
        feature_order : int
            Polynomial order: 1, 2, or 3 (only applicable for polynomial features)
        """
        self.env = env
        self.feature_type = feature_type
        self.feature_order = feature_order
        
    def get_feature_dim(self) -> int:
        """Get the dimensionality of feature vectors"""
        if self.feature_type == "one_hot":
            # One-hot: each state has its own dimension
            return self.env.n_states
        elif self.feature_type == "polynomial":
            if self.feature_order == 1:
                return 3  # Constant term + x + y
            elif self.feature_order == 2:
                return 6  # Constant + x + y + x² + xy + y²
            elif self.feature_order == 3:
                return 10  # Constant + x + y + x² + xy + y² + x³ + x²y + xy² + y³
            else:
                raise ValueError(f"Unsupported polynomial order: {self.feature_order}")
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
    
    def extract_features(self, state: int) -> np.ndarray:
        """
        Extract features for a given state
        
        Parameters:
        -----------
        state : int
            State index (0 to n_states-1)
            
        Returns:
        --------
        np.ndarray
            Feature vector representation of the state
        """
        if self.feature_type == "one_hot":
            return self._one_hot_features(state)
        elif self.feature_type == "polynomial":
            return self._polynomial_features(state)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")
    
    def _one_hot_features(self, state: int) -> np.ndarray:
        """Generate one-hot encoded features"""
        # Create zero vector with length equal to number of states
        features = np.zeros(self.env.n_states, dtype=np.float32)
        # Set the index corresponding to the state to 1.0
        features[state] = 1.0
        return features
    
    def _polynomial_features(self, state: int) -> np.ndarray:
        """Generate polynomial features"""
        # Convert state index to grid coordinates
        # Assuming grid is row-major: state = row * size + col
        row, col = state // self.env.size, state % self.env.size
        x, y = col, row
        
        # Normalize coordinates to range [-0.5, 0.5]
        # This helps with numerical stability and feature scaling
        x_norm = (x - (self.env.size - 1) * 0.5) / (self.env.size - 1)
        y_norm = (y - (self.env.size - 1) * 0.5) / (self.env.size - 1)
        
        # Generate polynomial features based on selected order
        if self.feature_order == 1:
            # Linear features: 1, x, y
            return np.array([1.0, x_norm, y_norm], dtype=np.float32)
        elif self.feature_order == 2:
            # Quadratic features: 1, x, y, x², xy, y²
            return np.array([1.0, x_norm, y_norm, 
                           x_norm**2, x_norm*y_norm, y_norm**2], dtype=np.float32)
        elif self.feature_order == 3:
            # Cubic features: 1, x, y, x², xy, y², x³, x²y, xy², y³
            return np.array([1.0, x_norm, y_norm, 
                           x_norm**2, x_norm*y_norm, y_norm**2,
                           x_norm**3, x_norm**2*y_norm, 
                           x_norm*y_norm**2, y_norm**3], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported polynomial order: {self.feature_order}")