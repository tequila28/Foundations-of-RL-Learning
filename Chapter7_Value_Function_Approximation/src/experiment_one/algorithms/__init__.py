"""
Value Function Approximation Algorithms
"""
import os
import sys


# Environment path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))


sys.path.append(current_dir)

from .policy_evaluator import EpisodeGenerator
from .feature_extractor import FeatureExtractor
from .td_linear import TDLinear
from .bellman_iteration import BellmanIteration

__all__ = [
    'EpisodeGenerator',
    'FeatureExtractor',
    'TDLinear',
    'BellmanIteration'
]