# truncated_policy_iteration.py
import numpy as np
import time
from typing import Dict
from dpsolver import DPSolver


class TruncatedPI(DPSolver):
    """Truncated Policy Iteration Algorithm"""

    def __init__(self, env, theta: float = 1e-6, max_iterations: int = 100, x: int = 5):
        super().__init__(env, theta, max_iterations)
        self.x = x

    def solve(self) -> Dict:
        """Main solving function for Truncated Policy Iteration"""
        history = {'values': [], 'iterations': 0, 'time': 0}
        start_time = time.time()
        
        # Use self.max_iterations or default to 100
        max_iters = self.max_iterations
        
        for i in range(max_iters):
            # Phase 1: Truncated Policy Evaluation
            # Only iterate x times, not until complete convergence
            for _ in range(self.x):
                for s in range(self.env.n_states):
                    a = self.policy[s]
                    self.V[s] = self.env.R[s, a] + self.env.gamma * np.dot(self.env.P[s, a], self.V)
            
            # Phase 2: Policy Improvement
            new_policy = np.array([np.argmax(self.get_q_values(s)) for s in range(self.env.n_states)])
            
            # Check if policy has converged
            if np.array_equal(new_policy, self.policy):
                break
            
            self.policy = new_policy
            history['values'].append(self.V.copy())
            
        history['iterations'] = i + 1
        history['time'] = time.time() - start_time
        return history