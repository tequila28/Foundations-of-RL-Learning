import numpy as np

class SGD:
    """
    Stochastic Gradient Descent (SGD) - Stochastic gradient descent optimizer
    
    This class implements the stochastic gradient descent algorithm, 
    which updates model parameters by computing gradients on a single
    randomly sampled data point at each iteration. This approach provides
    faster updates but with higher variance in the gradient estimates.
    """
    def __init__(self, environment):
        """
        Initialize the stochastic gradient descent optimizer.
        
        Parameters:
        environment: Environment object containing training data and true parameters.
        """
        self.env = environment
        
    def estimate(self, initial_point=np.array([50.0, 50.0]), 
                 alpha_type='1/k', constant_alpha=0.005):
        """
        Perform parameter estimation using stochastic gradient descent.
        
        Parameters:
        initial_point: Starting point for the optimization (default: [50, 50]).
        alpha_type: Learning rate schedule type. 
                   '1/k' for decreasing rate, or 'constant' for fixed rate.
        constant_alpha: Fixed learning rate value when alpha_type is not '1/k'.
        
        Returns:
        trajectory: Array of parameter estimates at each iteration.
        errors: Array of Euclidean errors (distance to true parameters) at each iteration.
        """
        w = initial_point.copy()
        trajectory = [w.copy()]
        errors = [np.linalg.norm(w - self.env.true_mean)]
        
        for k in range(1, self.env.total_iterations + 1):
            # Set learning rate based on schedule type
            alpha = 1.0 / k if alpha_type == '1/k' else constant_alpha
            
            # Random sampling with replacement (single data point)
            random_idx = np.random.randint(0, self.env.num_samples)
            x_k = self.env.samples[random_idx]
            
            # Update rule: w_{k+1} = w_k - α * (w_k - x_k)
            w = w - alpha * (w - x_k)
            
            # Record trajectory and error
            trajectory.append(w.copy())
            errors.append(np.linalg.norm(w - self.env.true_mean))
            
        return np.array(trajectory), np.array(errors)