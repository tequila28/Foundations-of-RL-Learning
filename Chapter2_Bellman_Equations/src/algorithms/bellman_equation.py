import numpy as np

class BellmanSolver:
    """Bellman Equation Solver for Markov Decision Processes"""
    
    def __init__(self, grid_world):
        """
        Initialize the solver with a grid world environment
        :param grid_world: Instantiated GridWorld environment object
        """
        self.grid_world = grid_world
        self.n_states = grid_world.n_states
    
    def closed_form_solution(self, policy):
        """
        Solve Bellman expectation equation using matrix inversion
        Formula: v_π = (I - γP_π)^(-1) r_π
        
        :param policy: Policy matrix of shape (n_states, n_actions)
        :return: Value function vector v_π
        """
        # Get policy-induced transition matrix and reward vector
        P_pi, r_pi = self.grid_world.get_policy_matrices(policy)
        
        I = np.eye(self.n_states)  # Identity matrix
        # Construct coefficient matrix: I - γP_π
        coefficient_matrix = I - self.grid_world.gamma * P_pi
        
        # Check matrix invertibility
        if np.linalg.matrix_rank(coefficient_matrix) < self.n_states:
            raise ValueError("Coefficient matrix is singular, cannot use closed-form solution")
        
        # Solve linear system: (I - γP_π)v = r_π
        v_pi = np.linalg.solve(coefficient_matrix, r_pi)
        return v_pi
    
    def iterative_solution(self, policy, max_iterations=1000, tolerance=1e-6):
        """
        Solve Bellman expectation equation using iterative method
        Formula: v_{k+1} = r_π + γP_π v_k
        
        :param policy: Policy matrix of shape (n_states, n_actions)
        :param max_iterations: Maximum number of iterations
        :param tolerance: Convergence threshold
        :return: Converged value function vector v_π
        """
        P_pi, r_pi = self.grid_world.get_policy_matrices(policy)
        
        # Initialize value function
        v = np.zeros(self.n_states)
        
        for iteration in range(max_iterations):
            # Bellman expectation equation iteration
            v_new = r_pi + self.grid_world.gamma * np.dot(P_pi, v)
            
            # Check convergence using maximum absolute difference
            if np.max(np.abs(v_new - v)) < tolerance:
                print(f"Converged after {iteration+1} iterations")
                break
            
            v = v_new.copy()
        else:
            print(f"Reached maximum iterations: {max_iterations}")
        
        return v