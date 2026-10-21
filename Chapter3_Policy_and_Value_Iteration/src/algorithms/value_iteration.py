# value_iteration.py
import numpy as np
import time
from typing import Dict

import os
import sys
# Get absolute path of current file (main.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from dpsolver import DPSolver

class ValueIteration(DPSolver):
    """Value Iteration Algorithm"""
    
    def solve(self) -> Dict:
        """Main solving function for Value Iteration"""
        history = {'values': [], 'iterations': 0, 'time': 0}
        start_time = time.time()
        
        max_iters = self.max_iterations
        
        for i in range(max_iters):
            delta = 0
            old_V = self.V.copy()
            
            # Phase 1: Value Update
            for s in range(self.env.n_states):
                q_values = self.get_q_values(s)
                self.V[s] = np.max(q_values)
                delta = max(delta, abs(old_V[s] - self.V[s]))
            
            history['values'].append(self.V.copy())
            
            # Check convergence
            if delta < self.theta:
                break
        
        # Phase 2: Policy Extraction
        for s in range(self.env.n_states):
            self.policy[s] = np.argmax(self.get_q_values(s))
            
        history['iterations'] = i + 1
        history['time'] = time.time() - start_time
        return history