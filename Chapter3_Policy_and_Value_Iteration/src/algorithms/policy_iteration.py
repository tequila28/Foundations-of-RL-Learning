# policy_iteration.py
import numpy as np
import time
from typing import Dict
from dpsolver import DPSolver


class PolicyIteration(DPSolver):
    """Policy Iteration Algorithm"""
    def solve(self) -> Dict:
            """Main solving function for Policy Iteration"""
            history = {'values': [], 'iterations': 0, 'time': 0}
            start_time = time.time()
            
            # Use self.max_iterations or default to 100
            max_iters = self.max_iterations if self.max_iterations is not None else 100
            
            for i in range(max_iters):
                # Phase 1: Policy Evaluation
                while True:
                    delta = 0
                    for s in range(self.env.n_states):
                        old_v = self.V[s]
                        # V(s) = R(s, pi(s)) + gamma * sum(P(s'|s, pi(s)) * V(s'))
                        a = self.policy[s]
                        self.V[s] = self.env.R[s, a] + self.env.gamma * np.dot(self.env.P[s, a], self.V)
                        delta = max(delta, abs(old_v - self.V[s]))
                    
                    # Check convergence of value function
                    if delta < self.theta:
                        break
                
                # Phase 2: Policy Improvement
                policy_stable = True
                for s in range(self.env.n_states):
                    old_action = self.policy[s]
                    self.policy[s] = np.argmax(self.get_q_values(s))
                    if old_action != self.policy[s]:
                        policy_stable = False
                
                history['values'].append(self.V.copy())
                
                # Check if policy is stable
                if policy_stable:
                    break
                    
            history['iterations'] = i + 1
            history['time'] = time.time() - start_time
            return history