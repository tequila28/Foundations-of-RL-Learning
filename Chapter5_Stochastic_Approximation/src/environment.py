import numpy as np

class Environment:
    def __init__(self, square_size=30, num_samples=400, total_iterations=400):
        """
        Initialize the simulation environment for gradient descent estimation experiments.
        
        This class creates a synthetic dataset and provides the true mean parameter
        for evaluating different gradient descent optimization methods.
        
        Parameters:
        -----------
        square_size : float, optional
            Size of the square region where data points are uniformly distributed.
            Points are generated within [-square_size/2, square_size/2] in both dimensions.
            Default is 30.
        num_samples : int, optional
            Total number of data points to generate. Default is 400.
        total_iterations : int, optional
            Number of iterations for gradient descent optimization. Default is 400.
        
        Attributes:
        -----------
        true_mean : numpy.ndarray
            The true mean parameter vector [0, 0] (ground truth).
        samples : numpy.ndarray or None
            Generated data samples of shape (num_samples, 2). Initialized as None.
        """
        self.square_size = square_size
        self.num_samples = num_samples
        self.total_iterations = total_iterations
        self.true_mean = np.array([0.0, 0.0])
        self.samples = None
        
    def generate_samples(self):
        """
        Generate uniformly distributed random samples within a square region.
        
        The samples are drawn from a uniform distribution in both x and y dimensions:
        x ~ Uniform(-square_size/2, square_size/2)
        y ~ Uniform(-square_size/2, square_size/2)
        
        Returns:
        --------
        numpy.ndarray
            Generated samples as a 2D array of shape (num_samples, 2).
            First column contains x-coordinates, second column contains y-coordinates.
        
        Notes:
        ------
        - Uses numpy.random.seed(42) for reproducibility.
        - Stores generated samples in self.samples attribute.
        - Each sample is a 2D point [x, y] where x, y ∈ [-square_size/2, square_size/2].
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Calculate half-size for symmetric distribution
        half_size = self.square_size / 2
        
        # Generate uniformly distributed samples
        # Each sample is a 2D point with coordinates in [-half_size, half_size]
        self.samples = np.random.uniform(
            low=-half_size, 
            high=half_size, 
            size=(self.num_samples, 2)
        )
        
        return self.samples