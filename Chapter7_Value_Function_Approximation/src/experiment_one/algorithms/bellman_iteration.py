import numpy as np

class BellmanIteration:
    """Bellman equation iteration algorithm for policy evaluation"""
    def __init__(self, env):
        """
        Initialize Bellman iteration algorithm
        
        Parameters:
        -----------
        env : GridWorld
            Grid world environment
        """
        self.env = env
        self.V = np.zeros(env.n_states)
        
    def iterate(self, max_iterations: int = 1000, threshold: float = 1e-6):
        """
        Perform synchronous policy evaluation
        Uses V_k from previous iteration to calculate V_{k+1}
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of iterations
        threshold : float
            Convergence threshold
            
        Returns:
        --------
        np.ndarray
            Converged state values
        """
        for i in range(max_iterations):
            V_new = np.zeros(self.env.n_states)  # Store new values
            V_prev = self.V.copy()  # Save values from previous iteration
            
            for state in range(self.env.n_states):
                value_sum = 0
                for action in self.env.actions:
                    # Get next state
                    next_state, hit_wall = self.env.transition_logic(state, action)
                    
                    # Handle wall collision
                    if hit_wall:
                        next_state = state
                    
                    # Get reward
                    reward = self.env.get_reward(next_state, hit_wall)
                    
                    value_sum += 0.2 * (reward + self.env.gamma * V_prev[next_state])
                
                V_new[state] = value_sum
            
            # Check convergence
            max_change = np.max(np.abs(V_new - V_prev))
            self.V = V_new  # Update all values at once
            
            if max_change < threshold:
                print(f"Converged at iteration {i+1}, max change: {max_change:.6f}")
                break
        
        return self.V