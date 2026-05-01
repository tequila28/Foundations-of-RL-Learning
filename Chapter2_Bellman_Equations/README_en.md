# Chapter 2: Bellman Equations

<div align="right">

[English](README_en.md) | [简体中文](README.md)

</div>

## Introduction

### Introduction to Bellman Equations
The Bellman Equation is a core equation in reinforcement learning and dynamic programming, used to describe the state-value function or action-value function under an optimal policy. It expresses the value of the current state as the sum of the current reward and the discounted value of the subsequent state, embodying the concepts of "optimal substructure" and "overlapping subproblems". This chapter implements two forms of the Bellman Equation (closed-form vector and iterative) and provides calculation and visualization in a Grid World environment.

### Implementation Content
This chapter implements the following in the Grid World environment:
1. Given one randomized policy and three deterministic policies, compute and visualize the state-value function V^π(s) for each policy.
2. Intuitively evaluate the quality of different policies by comparing their state values.
3. Grid World environment parameters (such as grid size, obstacle positions, reward function, discount factor, etc.) are manually adjustable.
4. Provides policy visualization (arrows represent action selection) and state-value visualization (color intensity represents value magnitude) functionalities.

Through interactive adjustments and visualization, users can more intuitively understand how the Bellman Equation evaluates strategies and observe the process of policy improvement.

## File Structure

```bash
Chapter2_Bellman_Equations/
├── results/ # Experimental results
│ ├── grid_world_policy_comparison_closed.png # Results from closed-form solution (vector form)
│ └── grid_world_policy_comparison_iterative.png # Results from iterative solution
├── src/ # Source code
│ ├── algorithms/ # Algorithm implementations
│ │ └── bellman_equation.py # Bellman equation implementation
│ ├── experiment.py # Main experiment file
│ └── visualization.py # Visualization utilities
└── scripts/ # Scripts directory
└── chapter2_experiment.sh # One-click experiment script
```

## Quick Start

```bash
bash Chapter2_Bellman_Equations/scripts/chapter2_experiment.sh
```

## Parameter Description

The following are the main parameters used in the experimental script and their meanings:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **SIZE** | 5 | Dimension of the square Grid World (number of rows and columns), creating a 5×5 grid |
| **GAMMA** | 0.9 | Discount factor for future rewards in reinforcement learning, ranging 0-1. Values closer to 1 indicate greater emphasis on future rewards |
| **FORBIDDEN_STATES** | "6 7 12 16 18 21" | States that incur penalties when entered |
| **TARGET_STATES** | "17" | List of target state indices. Reaching these states yields a reward |
| **R_BOUND** | -1 | Immediate reward received when an agent's action moves it outside the grid (hitting a wall), penalizing invalid moves |
| **R_FORBID** | -1 | Immediate reward received when an agent enters a forbidden state, penalizing entry into hazardous areas |
| **R_TARGET** | 1 | Immediate reward received when an agent reaches a target state, representing positive reward for achieving the goal |
| **R_DEFAULT** | 0 | Default immediate reward for any other valid transition (moving to a non-target, non-forbidden cell) |
| **SEED** | 42 | Seed used to initialize the random policy |

## Experimental Results
The experiment will generate two visualizations showing the state values for each policy solved under the respective Bellman equation form:

### Closed-Form Solution Visualization
![Random Policy Chart](./results/grid_world_policy_comparison_closed.png)

### Iterative Solution Visualization
![Policy Value Chart](./results/grid_world_policy_comparison_iterative.png)




