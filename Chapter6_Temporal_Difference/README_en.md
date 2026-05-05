# Chapter 6: Temporal Difference (TD) Algorithm Experiments

<div align="right">

[English](README_en.md) | [简体中文](README.md)

</div>

## Introduction

### **Basics of Temporal Difference Algorithms**

Temporal Difference (TD) methods are a core class of model-free reinforcement learning algorithms that combine ideas from Monte Carlo methods and dynamic programming. They learn through bootstrapping—updating current estimates based on other learned estimates—without requiring a complete model of the environment.

### **SARSA (State-Action-Reward-State-Action)**
- Standard on-policy TD control algorithm
- Follows ε-greedy exploration policy
- Updates value using actions selected by the current policy
- Suitable for online learning scenarios

### **Expected SARSA**
- Expectation-based TD control algorithm
- Updates using the expected value of all possible actions
- Generally more stable than standard SARSA
- Reduces variance and improves learning efficiency

### **N-step SARSA**
- Multi-step TD control algorithm
- Updates using n-step returns
- Balances immediate and long-term rewards
- Adjustable step parameter n controls bias-variance tradeoff

### **On-policy Q-Learning**
- Q-learning algorithm based on ε-greedy policy
- Behavior policy is the same as target policy
- Balances exploration and exploitation
- Suitable for environments requiring continuous exploration

### **Off-policy Q-Learning**
- Standard Q-learning algorithm
- Behavior policy is separated from target policy
- Learns optimal policy regardless of exploration strategy
- Theoretically guaranteed to converge to optimal policy

## Algorithm Implementation

This chapter implements the following five temporal difference algorithms in the Grid World environment to find optimal policies:

1.  **SARSA Algorithm Implementation**: Standard on-policy TD control algorithm
2.  **Expected SARSA Algorithm Implementation**: Expectation-based TD control algorithm
3.  **N-step SARSA Algorithm Implementation**: Multi-step TD control algorithm
4.  **On-policy Q-Learning Implementation**: ε-greedy policy-based Q-learning
5.  **Off-policy Q-Learning Implementation**: Standard Q-learning algorithm

## File Structure

```bash
Chapter6_Temporal_Difference/
├── results/ # Directory for experimental results
│ ├── final_policies_comparison.png # Final policy comparison diagram
│ └── multi_algorithm_comparison.png # Multi-algorithm training curve comparison diagram
├── scripts/ # Directory for experiment scripts
│ └── chapter6_experiment.sh # Main experiment script
└── src/ # Source code directory
├── algorithms/ # Algorithm implementation module
│ ├── expected_sarsa.py # Expected SARSA algorithm implementation
│ ├── n_step_sarsa.py # N-step SARSA algorithm implementation
│ ├── off_policy_qlearning.py # Off-policy Q-Learning implementation
│ ├── on_policy_qlearning.py # On-policy Q-Learning implementation
│ └── sarsa.py # SARSA algorithm implementation
├── experiment.py # Main file for experiment execution and parameter configuration
└── visualization.py # Data visualization and chart generation module
```

## Quick Start

```bash
bash Chapter6_Temporal_Difference/scripts/chapter6_experiment.sh
```

## Parameter Configuration

The following table describes the key parameters and their meanings used in the experiments:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| **GridWorld Environment Configuration** | | |
| **SIZE** | 5 | Grid World dimension, creating a 5×5 square grid |
| **GAMMA** | 0.9 | Discount factor for future rewards |
| **FORBIDDEN_STATES** | "6 7 12 16 18 21" | List of forbidden states |
| **TARGET_STATES** | "17" | List of target/terminal states |
| **R_BOUND** | -1 | Immediate reward received when hitting the grid boundary |
| **R_FORBID** | -1 | Immediate reward received when entering a forbidden state |
| **R_TARGET** | 1 | Immediate reward received when reaching a target state |
| **R_DEFAULT** | 0 | Default immediate reward for normal state transitions |
| **Algorithm Training Parameters** | | |
| **NUM_EPISODES** | 100 | Total number of training episodes |
| **MAX_STEPS** | 1000 | Maximum number of steps allowed per episode |
| **Algorithm Hyperparameters** | | |
| **LEARNING_RATE** | 0.1 | Learning rate parameter, controlling the magnitude of each update |
| **EPSILON** | 0.2 | Initial exploration rate parameter (ε-greedy policy) |
| **EPSILON_DECAY** | 0.998 | Exploration rate decay coefficient, decay ratio per episode |
| **EPSILON_MIN** | 0.01 | Minimum value of exploration rate |
| **N-step SARSA Parameters** | | |
| **N_STEPS** | "1 10" | Step parameter list for N-step SARSA algorithm (can test two step sizes) |

## Experimental Results

The experiments will generate two comprehensive visualization results, comparing and analyzing the learning performance of the five TD algorithms from multiple dimensions.

### 1. Multi-Algorithm Policy and State Value Comparison Diagram (6 subplots)
Each subplot shows the learning result of a TD algorithm in a 5×5 Grid World, including:
- **Grid Structure**: Clearly displays the 5×5 grid layout
- **State Value Labels**: Displays the estimated value of each state (in numerical format) at the center of each cell
- **Policy Arrows**: Indicates the learned policy for each cell through arrow direction (up/down/left/right/stay)
- **Special State Markers**:
  - Target states: Displayed in blue
  - Forbidden states: Displayed in orange

**Specifically includes the following 6 subplots**:
1. **SARSA Algorithm Results**: Standard on-policy TD learning
2. **Expected SARSA Algorithm Results**: Expectation-based TD learning
3. **N-step SARSA (n=1) Algorithm Results**: Single-step temporal difference
4. **N-step SARSA (n=10) Algorithm Results**: Ten-step temporal difference
5. **On-policy Q-Learning Results**: ε-greedy policy learning
6. **Off-policy Q-Learning Results**: Optimal policy learning

*This diagram intuitively compares the specific policy choices and state value estimates learned by different algorithms in the same environment.*

### 2. Multi-Algorithm Training Process Comparison Diagram (2 subplots)
- **Left Subplot: Total Reward Convergence Curve**
  - Shows the total reward obtained by the five algorithms in each training episode
  - Includes moving average lines and smoothing
  - Compares convergence speed and final performance of different algorithms

- **Right Subplot: TD Error Convergence Curve**
  - Shows the average TD error of the five algorithms in each training episode
  - Reflects the improvement process of the algorithms' value function estimation
  - Analyzes the learning stability of different algorithms

### Multi-Algorithm Optimal Policy and State Value Comparison
Shows the final policies and their corresponding state value functions learned by the five TD algorithms under the same training conditions:
![TD Algorithm State Value Comparison](./results/final_policies_comparison.png)

### Multi-Algorithm Training Process Comparison
Shows the performance indicator changes of the five TD algorithms during training:
![TD Algorithm Training Curve Comparison](./results/multi_algorithm_comparison.png)
