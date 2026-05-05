import numpy as np
from typing import Tuple

class StateFeatureExtractor:
    """State feature extractor for grid world using polynomial features only"""
    
    def __init__(self, grid_size: int, poly_order: int = 1):
        """
        Initialize feature extractor for states only with polynomial features
        
        Parameters:
        -----------
        grid_size : int
            Size of the grid (grid_size x grid_size)
        poly_order : int, default=1
            Polynomial order to use (1, 2, or 3)
        """
        self.grid_size = grid_size
        
        # Validate polynomial order
        if poly_order not in [1, 2, 3]:
            raise ValueError("poly_order must be 1, 2, or 3")
        
        self.poly_order = poly_order
        
        # Calculate state features dimension based on polynomial order
        if poly_order == 1:
            self.state_features_dim = 3  # 1 + x + y
        elif poly_order == 2:
            self.state_features_dim = 6  # 1 + x + y + x² + xy + y²
        elif poly_order == 3:
            self.state_features_dim = 10  # 1 + x + y + x² + xy + y² + x³ + x²y + xy² + y³
    
    def _get_coordinates(self, state: int) -> Tuple[int, int]:
        """Convert state index to grid coordinates"""
        row = state // self.grid_size
        col = state % self.grid_size
        return row, col
    
    def _normalize_coordinates(self, row: int, col: int) -> Tuple[float, float]:
        """Normalize coordinates to [-0.5, 0.5] range"""
        if self.grid_size <= 1:
            return 0.0, 0.0
        
        x_norm = (col - (self.grid_size - 1) * 0.5) / (self.grid_size - 1)
        y_norm = (row - (self.grid_size - 1) * 0.5) / (self.grid_size - 1)
        return x_norm, y_norm
    
    def extract_state_features(self, state: int) -> np.ndarray:
        """
        Extract polynomial features for a single state
        
        Returns:
        --------
        np.ndarray
            State feature vector containing polynomial features of specified order
        """
        row, col = self._get_coordinates(state)
        x_norm, y_norm = self._normalize_coordinates(row, col)
        
        if self.poly_order == 1:
            # 1st order polynomial: 1, x, y
            return np.array([1.0, x_norm, y_norm])
        
        elif self.poly_order == 2:
            # 2nd order polynomial: 1, x, y, x², xy, y²
            return np.array([
                1.0, x_norm, y_norm,
                x_norm**2, x_norm*y_norm, y_norm**2
            ])
        
        elif self.poly_order == 3:
            # 3rd order polynomial: 1, x, y, x², xy, y², x³, x²y, xy², y³
            return np.array([
                1.0, x_norm, y_norm,
                x_norm**2, x_norm*y_norm, y_norm**2,
                x_norm**3, x_norm**2*y_norm, x_norm*y_norm**2, y_norm**3
            ])
    
    def extract_state_features_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Extract features for a batch of states
        
        Parameters:
        -----------
        states : np.ndarray
            Array of state indices
            
        Returns:
        --------
        np.ndarray
            Batch of state feature vectors
        """
        batch_size = len(states)
        features = np.zeros((batch_size, self.state_features_dim))
        
        for i, state in enumerate(states):
            features[i] = self.extract_state_features(state)
            
        return features
    
    def get_feature_dimension(self) -> int:
        """
        Get the dimension of state feature vectors
        
        Returns:
        --------
        int
            Number of features in state vector
        """
        return self.state_features_dim
    
    def get_polynomial_terms(self) -> list:
        """
        Get the polynomial terms included in the feature vector
        
        Returns:
        --------
        list
            List of polynomial term names
        """
        if self.poly_order == 1:
            return ["1", "x", "y"]
        elif self.poly_order == 2:
            return ["1", "x", "y", "x²", "xy", "y²"]
        elif self.poly_order == 3:
            return ["1", "x", "y", "x²", "xy", "y²", "x³", "x²y", "xy²", "y³"]
    
    def __str__(self) -> str:
        """String representation of the feature extractor"""
        terms = self.get_polynomial_terms()
        return (f"StateFeatureExtractor(grid_size={self.grid_size}, "
                f"poly_order={self.poly_order}, "
                f"feature_dim={self.state_features_dim}, "
                f"terms={terms})")