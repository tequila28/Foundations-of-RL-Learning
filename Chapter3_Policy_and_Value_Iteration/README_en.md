# Chapter 3: Value Iteration and Policy Iteration Methods

<div align="right">

[English](README_en.md) | [简体中文](README.md)

</div>

## Introduction

### **Value Iteration Algorithm**

This algorithm iteratively updates the state value function and extracts the optimal policy in one step after convergence. Its core is the direct application of the Bellman optimality equation.

### **Policy Iteration Algorithm**

This algorithm alternates between two steps: "policy evaluation" and "policy improvement". It first fixes a policy and evaluates its value function, then greedily improves the policy based on that value function until the policy converges.

### **Truncated Policy Iteration**

This algorithm is an efficient variant of policy iteration. In the "policy evaluation" step, it does not pursue a fully converged value function. Instead, it performs only a fixed number of iterations (i.e., "truncation") before directly moving to the "policy improvement" step, thereby balancing computational efficiency and accuracy.

### **Implementation Content**

This chapter implements the following in a Grid World environment:

1.  **Implementation of Three Iteration Algorithms**: Full implementations of the value iteration, policy iteration, and truncated policy iteration algorithms.
2.  **Optimal Policy Solution**: Application of the three algorithms to solve for the optimal policy in the grid world environment.
3.  **State Value Visualization**: Visual representation of the state value function corresponding to the optimal policy found by each algorithm.
4.  **Flexible Environment Configuration**: The grid world environment parameters (such as grid size, obstacle positions, reward/penalty values, discount factor, terminal states, etc.) support flexible configuration.
5.  **Comparative Analysis**: Comparative analysis of the three algorithms in terms of convergence speed, computational efficiency, and final policy quality through visualized results.

## File Structure

```bash
Chapter3_Policy_and_Value_Iteration/ # Main directory for Chapter 3
├── results/ # Directory for experimental results
│ ├── convergence_comparison_S0.png # Convergence curve comparison for state S0 across the three algorithms
│ ├── policy_comparison.png # Visual comparison of optimal policies found by the three algorithms
│ └── TPI_error_vs_x.png # Relationship between error and truncation parameter x in truncated policy iteration
├── scripts/ # Scripts directory
│ └── chapter3_experiment.sh # Script to run all experiments with one command
└── src/ # Source code directory
├── algorithms/ # Algorithm implementations
│ ├── dpsolver.py # Base class for dynamic programming solvers
│ ├── policy_iteration.py # Policy iteration algorithm implementation
│ ├── truncated_policy_iteration.py # Truncated policy iteration algorithm implementation
│ └── value_iteration.py # Value iteration algorithm implementation
├── experiment.py # Main program for experiment configuration and execution
└── visualization.py # Visualization tools (policy, state values, convergence curves, etc.)
```

## Quick Start

```bash
bash Chapter3_Policy_and_Value_Iteration/scripts/chapter3_experiment.sh
```

## Experimental Results

The experiment will generate three visualization graphs: the change curve between the number of convergence iterations and the state value for the three algorithms, the iteration count and error curve for the truncated policy iteration algorithm with different numbers of inner loops, and the best policies corresponding to the three algorithms:

### Visualization of Change Curve Between Convergence Iterations and State Value
![Change Curve Between Convergence Iterations and State Value](./results/convergence_comparison_S0.png)

### Visualization of Iteration Counts for Convergence of Truncated Policy Iteration Algorithm with Different Inner Loop Counts
![Truncated Policy Iteration Algorithm with Different Inner Loop Counts](./results/TPI_error_vs_x.png)

### Visualization of Best Policies and Corresponding State Values for the Three Algorithms
![Best Policies for the Three Algorithms](./results/policy_comparison.png)
