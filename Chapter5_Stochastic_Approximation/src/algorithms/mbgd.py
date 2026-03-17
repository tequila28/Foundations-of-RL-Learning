import numpy as np

class MBGD:
    """
    Mini-batch Gradient Descent (MBGD) - Mini-batch gradient descent optimizer
    
    This class implements the mini-batch gradient descent algorithm, 
    which updates model parameters by computing gradients on a subset (mini-batch)
    of training samples. This approach balances efficiency and variance.
    """
    def __init__(self, environment):
        """
        Initialize the mini-batch gradient descent optimizer.
        
        Parameters:
        environment: Environment object containing training data and true parameters.
        """
        self.env = environment
        
    def estimate(self, initial_point=np.array([50.0, 50.0]), 
                 batch_size=10, alpha_type='1/k', constant_alpha=0.005):
        """
        Perform parameter estimation using mini-batch gradient descent.
        
        Parameters:
        initial_point: Starting point for the optimization (default: [50, 50]).
        batch_size: Size of each mini-batch for gradient computation.
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
            
            # Random mini-batch sampling with replacement
            batch_indices = np.random.randint(0, self.env.num_samples, batch_size)
            batch_samples = self.env.samples[batch_indices]
            batch_mean = np.mean(batch_samples, axis=0)
            
            # Update rule: w = w - α * (w - batch_mean)
            w = w - alpha * (w - batch_mean)
            
            # Record trajectory and error
            trajectory.append(w.copy())
            errors.append(np.linalg.norm(w - self.env.true_mean))
            
        return np.array(trajectory), np.array(errors)