# Chapter 7: Value Function Approximation Experiments

This chapter implements and compares several value function learning and control methods based on function approximation (linear feature TD, feature-based Q-learning / SARSA, etc.), and visualizes the state values and policies learned by each method in a Grid World environment. It supports configurable environments, feature extractors, and visual output, facilitating multi-algorithm comparison and training curve analysis.

<div align="right">

[English](README_en.md) | [简体中文](README.md)

</div>

## Introduction

### **Fundamentals of Value Function Approximation**

In scenarios with large or continuous state spaces, traditional tabular reinforcement learning methods (such as the TD algorithms from previous chapters) fail due to insufficient storage and generalization capabilities. Value Function Approximation (VFA) is a core technique to address this problem. It uses parameterized functions (e.g., linear functions, neural networks) to approximate the state-value function V(s) or the action-value function Q(s, a). This chapter focuses on approximation methods based on linear features, which map high-dimensional states to a low-dimensional feature space and learn the value function by updating weights, thereby achieving generalization for large-scale or continuous state spaces.

### **Linear Temporal Difference (Linear TD) Algorithm**
- Uses linear features in TD(0) and TD(λ) methods
- Approximates state values via the dot product of feature vectors and weight vectors
- Updates weights instead of each individual state value, enabling efficient learning
- Suitable for policy evaluation in large state spaces

### **Feature-based Q-Learning**
- Combines the Q-Learning algorithm with linear function approximation
- Represents state-action pairs via features to learn the optimal action-value function
- Uses an ε-greedy policy for exploration
- Suitable for control problems in high-dimensional discrete or low-dimensional continuous state spaces

### **Feature-based SARSA**
- Combines the on-policy SARSA algorithm with linear function approximation
- Uses the action selected by the current policy for Q-value updates
- Directly optimizes the behavior policy during learning
- Performs stably in online control scenarios requiring a balance between exploration and exploitation

### **Feature Extractor**
- A core module responsible for converting raw states (e.g., grid coordinates) into feature vectors
- Implements **Polynomial Features** to capture interactions between state components

## File Structure
```bash
Chapter7_Value_Function_Approximation/
├── experiment_one_results/ # Experiment One Result Storage Directory
│ ├── 3d_state_values.png # 3D State Value Visualization
│ ├── comparison_summary.png # Comprehensive Comparison Chart
│ ├── ground_truth_3d.png # Ground Truth 3D Visualization
│ └── rmse_curves_500_episodes.png # RMSE Error Curve Plot
│
├── experiment_two_results/ # Experiment Two Result Storage Directory
│ ├── final_policies_comparison.png # Final Policy Comparison Chart
│ └── multi_algorithm_comparison.png # Multi-Algorithm Training Comparison Chart
│
├── scripts/ # Scripts Directory
│ ├── chapter7_experiment_one.sh # Experiment One Execution Script
│ └── chapter7_experiment_two.sh # Experiment Two Execution Script
│
├── src/ # Source Code Directory
│ ├── experiment_one/ # Experiment One Module
│ │ ├── algorithms/ # Algorithms Subdirectory
│ │ │ ├── init.py
│ │ │ ├── bellman_iteration.py
│ │ │ ├── feature_extractor.py
│ │ │ ├── policy_generator.py
│ │ │ └── td_linear.py
│ │ ├── experiment.py # Experiment One Main File for Running and Parameter Configuration
│ │ └── visualization.py # Experiment One Data Visualization and Plot Generation Module
│ │
│ └── experiment_two/ # Experiment Two Module
│ ├── algorithms/ # Algorithms Module
│ │ ├── init.py
│ │ ├── feature_extractor.py
│ │ ├── qlearning_agent.py
│ │ └── sarsa_agent.py
│ ├── experiment.py # Experiment Two Main File for Running and Parameter Configuration
│ └── visualization.py # Experiment Two Data Visualization and Plot Generation Module
│
└── README.md # Project Documentation
```
## Quick Start

### Experiment One
Run Experiment One to compare the value function approximation performance of linear TD under a given random policy:
```bash
bash Chapter7_Value_Function_Approximation/scripts/chapter7_experiment_one.sh
```
Below are the key parameters used in the experiment and their meanings:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| **GridWorld Environment Configuration** | | |
| **SIZE** | 5 | Grid world dimension, creates a 5×5 square grid |
| **GAMMA** | 0.9 | Discount factor for future rewards, controls importance of future rewards |
| **FORBIDDEN_STATES** | "6 7 12 16 18 21" | List of forbidden states (counting from 0) |
| **TARGET_STATES** | "17" | List of target/terminal states, episode ends upon reaching them |
| **R_BOUND** | -1 | Immediate reward received when hitting the grid boundary |
| **R_FORBID** | -1 | Immediate reward received when entering a forbidden state |
| **R_TARGET** | 1 | Immediate reward received when reaching a target state |
| **R_DEFAULT** | 0 | Default immediate reward for normal state transitions |
| **Experiment One Algorithm Training Parameters** | | |
| **N_EPISODES** | 500 | Total number of training episodes |
| **LEARNING_RATE** | 0.001 | Learning rate parameter, controls the magnitude of each update |
| **SEED** | 42 | Random seed to ensure reproducible experiments |

### Experiment Two
Run Experiment Two to compare the online learning performance of feature-based Q-Learning and SARSA in the grid world:
```bash
bash Chapter7_Value_Function_Approximation/scripts/chapter7_experiment_two.sh
```
Below are the key parameters used in Experiment Two and their meanings:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| **GridWorld Environment Configuration** | | |
| **SIZE** | 5 | Grid world dimension, creates a 5×5 square grid |
| **GAMMA** | 0.9 | Discount factor for future rewards, controls importance of future rewards |
| **FORBIDDEN_STATES** | "6 7 12 16 18 21" | List of forbidden states (counting from 0) |
| **TARGET_STATES** | "17" | List of target/terminal states, episode ends upon reaching them |
| **R_BOUND** | -1 | Immediate reward received when hitting the grid boundary |
| **R_FORBID** | -1 | Immediate reward received when entering a forbidden state |
| **R_TARGET** | 1 | Immediate reward received when reaching a target state |
| **R_DEFAULT** | 0 | Default immediate reward for normal state transitions |
| **Algorithm Training Parameters** | | |
| **NUM_EPISODES** | 1000 | Total number of training episodes |
| **MAX_STEPS** | 100 | Maximum step limit per episode |
| **Algorithm Hyperparameters** | | |
| **LEARNING_RATE** | 0.0005 | Learning rate parameter, controls the magnitude of each update |
| **EPSILON** | 0.2 | Initial exploration rate parameter (for ε-greedy policy) |
| **EPSILON_DECAY** | 0.998 | Exploration rate decay factor, decay ratio per episode |
| **EPSILON_MIN** | 0.01 | Minimum value for the exploration rate |

## Experimental Results

### Experiment One Results

Given a fixed random policy, uses linear function approximation to evaluate the state-value function and compares it with the true values (computed via the Bellman equation). Experiment One will generate the following visual results to analyze the approximation performance of the linear TD algorithm:

#### 1. Comprehensive Comparison Chart
Shows the comparison between state values learned by linear TD(0) and TD(λ) and the true values:
![Comparison Summary](./experiment_one_results/comparison_summary.png)

#### 2. 3D State Value Visualization
Three-dimensional visualization of the state value distributions learned by different methods:
- **Ground Truth 3D Chart**: Precise state values calculated via the Bellman equation
![Ground Truth 3D](./experiment_one_results/ground_truth_3d.png)
- **Approximated Values 3D Chart**: State values learned by the linear TD algorithm
![3D State Values](./experiment_one_results/3d_state_values.png)

#### 3. RMSE Error Curve
Shows the change in approximation error during training, comparing the convergence speed of different algorithms:
![RMSE Curve](./experiment_one_results/rmse_curves_500_episodes.png)

### Experiment Two Results
**Note ⚠️**: The current online control algorithms based on linear features have convergence issues when learning optimal policies in the grid world environment; the learned policies may not be optimal. This is primarily due to the limited expressive power of linear function approximation. In discrete environments like grid worlds, tabular methods might be more suitable. Feature selection and approximation errors can lead to insufficient policy learning.
Learns optimal policies by interacting online with an unknown environment dynamic using algorithms based on linear feature approximation. Experiment Two will generate the following visual results to compare feature-based online control algorithms:

#### 1. Multi-Algorithm Policy and State Value Comparison
Shows the final policies learned by Q-Learning and SARSA and their corresponding state-value functions:
![Final Policy Comparison](./experiment_two_results/final_policies_comparison.png)

#### 2. Multi-Algorithm Training Process Comparison
Shows the changes in performance metrics of the two algorithms during training:
- **Left Subplot**: Convergence curve of total reward per episode
- **Right Subplot**: TD error convergence curve during training
![Training Curve Comparison](./experiment_two_results/multi_algorithm_comparison.png)
