# dpsolver.py
import numpy as np
from typing import Dict, Optional


class DPSolver:
    """Dynamic Programming Solver Base Class"""
    def __init__(self, env, theta: float = 1e-6, max_iterations: Optional[int] = 1000):
        """
        Initialize the DP Solver with configurable parameters.
        
        Args:
            env: The environment instance
            theta: Convergence threshold (delta)
            max_iterations: Maximum number of iterations allowed
        """
        self.env = env
        self.V = np.zeros(env.n_states)
        # Here uniformly initialized to 0, will be updated by algorithm
        self.policy = np.zeros(env.n_states, dtype=int)
        self.theta = theta  # Convergence threshold
        self.max_iterations = max_iterations  # Maximum iterations

    def get_q_values(self, state: int) -> np.ndarray:
        """
        Calculate Q-values for all actions in current state according to Bellman equation:
        Q(s, a) = R(s, a) + gamma * sum(P(s'|s, a) * V(s'))
        """
        return self.env.R[state] + self.env.gamma * np.dot(self.env.P[state], self.V)