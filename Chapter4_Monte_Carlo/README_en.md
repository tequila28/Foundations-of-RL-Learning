# Chapter 4: Monte Carlo Algorithm Experiments

<div align="right">

[English](README_en.md) | [简体中文](README.md)

</div>

## Introduction

### **Basics of Monte Carlo Algorithms**

Monte Carlo methods estimate state values and optimize policies by sampling complete trajectories (episodes) from the environment. Unlike dynamic programming-based algorithms, they do not require complete knowledge of the environment model, relying solely on interactive experience with the environment.

### **MC Basic (Basic Monte Carlo Control)**
- Based on every-visit Monte Carlo updates
- No ε-greedy exploration mechanism
- Fixed policy evaluation

### **MC Exploring Starts**
- Each episode starts from a random state-action pair
- Ensures thorough exploration of the state-action space
- Uses greedy policy improvement

### **MC ε-greedy**
- ε-greedy strategy balancing exploration and exploitation
- Adjustable exploration rate ε
- Incremental policy improvement

## Implementation Content

This chapter implements the following in the Grid World environment:

1.  **Three Algorithm Implementations**: Complete implementations of MC Basic, MC Exploring Starts, and MC ε-greedy algorithms
2.  **Policy and Value Visualization**: Visualizes the final learned policy and its corresponding state value function for each method in the Grid World
3.  **Configurable Environment Model**: Flexible configuration of Grid World parameters (grid size, obstacle positions, reward values, terminal states, etc.)
4.  **ε-Value Comparison Analysis**: Visualizes learning curves and performance scatter plots for the MC ε-greedy algorithm with different ε values, analyzing the exploration-exploitation trade-off
5.  **Trajectory Sampling and Analysis**: Displays complete trajectories sampled by the algorithms, analyzing the impact of different sampling strategies on learning effectiveness
6.  **Convergence Comparison**: Compares the convergence speed and final policy quality of the three Monte Carlo methods

## File Structure

```bash
Chapter4_Monte_Carlo/
├── results/                          # Directory for result visualization files
│   ├── mc_policy_comparison.png      # Policy comparison of three algorithms
│   └── state_action_visit_scatter.png # Exploration characteristics analysis scatter plot
├── scripts/                          # Directory for experiment scripts
│   └── chapter4_experiment.sh        # Main experiment script, run all experiments with one click
└── src/                              # Source code directory
    ├── algorithms/                   # Algorithm implementation module
    │   ├── mc_basic.py               # MC Basic algorithm implementation
    │   ├── mc_epsilon_greedy.py      # MC ε-greedy algorithm implementation
    │   └── mc_exploring_starts.py    # MC Exploring Starts algorithm implementation
    ├── experiment.py                 # Main file for experiment execution and parameter configuration
    └── visualization.py              # Data visualization and chart generation module
```

## Quick Start

```bash
bash Chapter4_Monte_Carlo/scripts/chapter4_experiment.sh
```

## Parameter Configuration

The following table describes the parameters and their default values used in the experiments:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **GridWorld Environment Configuration** | | |
| **SIZE** | 5 | Grid World dimension, creating a 5×5 square grid |
| **GAMMA** | 0.9 | Discount factor for future rewards, ranging 0-1. Higher values place more importance on future rewards |
| **FORBIDDEN_STATES** | "6 7 12 16 18 21" | List of forbidden states |
| **TARGET_STATES** | "17" | List of target/terminal states. Episodes end upon reaching these states |
| **R_BOUND** | -1 | Immediate reward received when hitting the grid boundary |
| **R_FORBID** | -1 | Immediate reward received when entering a forbidden state |
| **R_TARGET** | 1 | Immediate reward received when reaching a target state |
| **R_DEFAULT** | 0 | Default immediate reward for normal state transitions |
| **MC Basic Algorithm Configuration** | | |
| **MC_BASIC_EPISODE_LENGTH** | 10 | Maximum number of steps allowed per episode |
| **MC_BASIC_ITERATIONS** | 100 | Total number of training iterations for the algorithm |
| **MC Exploring Starts Algorithm Configuration** | | |
| **MC_ES_EPISODE_LENGTH** | 10 | Maximum number of steps allowed per episode |
| **MC_ES_ITERATIONS** | 10000 | Total number of training iterations for the algorithm |
| **MC ε-greedy Algorithm Configuration (ε=0.1)** | | |
| **MC_EPS1_EPISODE_LENGTH** | 1000 | Maximum number of steps allowed per episode |
| **MC_EPS1_EPSILON** | 0.1 | Exploration rate parameter, balancing exploration and exploitation |
| **MC_EPS1_ITERATIONS** | 100 | Total number of training iterations for the algorithm |
| **MC ε-greedy Algorithm Configuration (ε=0.2)** | | |
| **MC_EPS2_EPISODE_LENGTH** | 1000 | Maximum number of steps allowed per episode |
| **MC_EPS2_EPSILON** | 0.2 | Exploration rate parameter, balancing exploration and exploitation |
| **MC_EPS2_ITERATIONS** | 100 | Total number of training iterations for the algorithm |
| **Exploration Characteristics Analysis Configuration** | | |
| **MC_EPISODE_LENGTH** | 100000 | Total number of steps used for scatter plot analysis |
| **MC_EPS_EPSILON1** | 1.0 | First exploration rate parameter (full exploration) |
| **MC_EPS_EPSILON2** | 0.2 | Second exploration rate parameter (partial exploration) |

## Experimental Results

The experiments will present **two types of comparative analysis visualizations** to comprehensively demonstrate the learning effectiveness and exploration characteristics of the three algorithms.

1.  **Policy Comparison Visualization** (Four subplots in total)
    -   Final policy learned by the **MC Basic** algorithm
    -   Final policy learned by the **MC Exploring Starts** algorithm
    -   Final policy learned by the **MC ε-greedy** algorithm (parameter ε1)
    -   Final policy learned by the **MC ε-greedy** algorithm (parameter ε2)

    *The visualization presents the final action selection policies learned by different algorithms in the same Grid World, facilitating intuitive comparison.*

2.  **Exploration Characteristics Analysis Visualization**
    -   Shows the distribution of visits to each state-action pair during exploration for the MC ε-greedy algorithm under different ε values
    -   Helps analyze the impact of exploration strategies on algorithm performance and learning efficiency

### Comparison of Optimal Policies and State Values for the Three Algorithms
![Monte Carlo Algorithm State Value Comparison](./results/mc_policy_comparison.png)

### Scatter Plot of State-Action Pair Visit Frequency Under Different ε Values
- Shows exploration behavior differences under ε=1.0 (full exploration) and ε=0.2 (partial exploration)
- X-axis: State-action pair index
- Y-axis: Number of visits
![State-Action Pair Visit Frequency Scatter Plot](./results/state_action_visit_scatter.png)
