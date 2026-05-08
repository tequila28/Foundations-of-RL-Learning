import torch
import torch.nn as nn
from typing import List, Optional


class PolicyNetwork(nn.Module):
    """Policy network: maps states to action probability distributions."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_size: int = 128,
                 hidden_layers: Optional[List[int]] = None):
        """
        Initialize the policy network.

        Parameters:
        -----------
        state_dim : int
            State dimension
        action_dim : int
            Action dimension
        hidden_size : int
            Hidden layer size (used when hidden_layers is None)
        hidden_layers : Optional[List[int]]
            List of hidden layer sizes, e.g., [128, 64] for two hidden layers
        """
        super(PolicyNetwork, self).__init__()

        
        hidden_layers = [hidden_size]

        # Build network layers
        layers = []

        # Input layer to first hidden layer
        layers.append(nn.Linear(state_dim, hidden_layers[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], action_dim))
        layers.append(nn.Softmax(dim=-1))

        # Create the network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Parameters:
        -----------
        x : torch.Tensor
            Input state vector

        Returns:
        --------
        torch.Tensor
            Action probability distribution
        """
        return self.network(x)


class ValueNetwork(nn.Module):
    """Value function network: maps states to state values."""

    def __init__(self,
                 state_dim: int,
                 hidden_size: int = 128,
                 hidden_layers: Optional[List[int]] = None):
        """
        Initialize the value function network.

        Parameters:
        -----------
        state_dim : int
            State dimension
        hidden_size : int
            Hidden layer size (used when hidden_layers is None)
        hidden_layers : Optional[List[int]]
            List of hidden layer sizes, e.g., [128, 64] for two hidden layers
        """
        super(ValueNetwork, self).__init__()

        # Default hidden layer structure
        if hidden_layers is None:
            hidden_layers = [hidden_size]

        # Build network layers
        layers = []

        # Input layer to first hidden layer
        layers.append(nn.Linear(state_dim, hidden_layers[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())

        # Output layer (value function output, no activation)
        layers.append(nn.Linear(hidden_layers[-1], 1))

        # Create the network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Parameters:
        -----------
        x : torch.Tensor
            Input state vector

        Returns:
        --------
        torch.Tensor
            State value estimate
        """
        return self.network(x)