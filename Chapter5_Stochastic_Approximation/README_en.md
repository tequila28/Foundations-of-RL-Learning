# Chapter 5: Stochastic Approximation Algorithms

<div align="right">

[English](README_en.md) | [简体中文](README.md)

</div>

## Introduction

### **Basics of Stochastic Approximation Algorithms**

Stochastic Approximation is a powerful method for handling stochastic optimization problems, particularly well-suited for reinforcement learning environments with uncertainty or random noise. Unlike deterministic optimization methods, it processes imprecise gradient information by introducing random sampling, enabling efficient optimization in complex environments.

### **Batch Gradient Descent (BGD)**
- Uses all training samples to compute gradients in each iteration
- Stable gradient update direction, smooth convergence path
- High computational cost and memory requirements
- Suitable for small-scale datasets

### **Stochastic Gradient Descent (SGD)**
- Uses a single sample to compute gradients in each iteration
- High variance in gradient updates, oscillating convergence path
- High computational efficiency, low memory requirements
- Suitable for large-scale online learning scenarios

### **Mini-Batch Gradient Descent (MBGD)**
- Uses a small batch of samples to compute gradients in each iteration
- Balances between the efficiency of SGD and the stability of BGD
- Enables parallel computation, fully utilizing hardware resources
- Widely used in deep learning practice

## Algorithm Implementation

This chapter implements the following three stochastic approximation methods in a 2D scatter plot mean estimation problem:

1.  **BGD Algorithm Implementation**: Complete batch gradient descent implementation showcasing deterministic optimization processes
2.  **SGD Algorithm Implementation**: Stochastic gradient descent algorithm highlighting characteristics of high-variance gradient updates
3.  **MBGD Algorithm Implementation**: Mini-batch gradient descent algorithm balancing convergence speed and stability

## File Structure

```bash
Chapter5_Stochastic_Approximation/
├── results/ # Directory for main experimental results
│ ├── BGD_results.png # Batch Gradient Descent algorithm results
│ ├── MBGD_results.png # Mini-Batch Gradient Descent algorithm results
│ └── SGD_results.png # Stochastic Gradient Descent algorithm results
├── scripts/ # Experimental scripts
│ └── chapter5_experiment.sh # Script for running experiments
├── src/ # Source code directory
│ ├── environment.py # Environment definition
│ ├── experiment.py # Experiment logic implementation
│ ├── visualization.py # Visualization tools
│ ├── algorithms/ # Algorithm implementation directory
│ │ ├── bgd.py # Batch Gradient Descent algorithm
│ │ ├── mbgd.py # Mini-Batch Gradient Descent algorithm
│ │ └── sgd.py # Stochastic Gradient Descent algorithm
└── README.md # Project documentation
```

## Quick Start

### Run Command

```bash
bash Chapter5_Stochastic_Approximation/scripts/chapter5_experiment.sh
```

## Parameter Configuration

The following table describes the key parameters and their meanings used in the experiments:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| **Experimental Environment Parameters** | | |
| **SQUARE_SIZE** | 50.0 | Side length of the square for uniform distribution |
| **NUM_SAMPLES** | 1000 | Total number of samples generated |
| **TOTAL_ITERATIONS** | 400 | Total number of optimization iterations |
| **INIT_X** | 50.0 | Initial x-coordinate of the starting point |
| **INIT_Y** | 50.0 | Initial y-coordinate of the starting point |
| **Gradient Descent Parameters** | | |
| **BATCH_SIZES** | "10 50" | Batch sizes for Mini-Batch Gradient Descent (multiple can be specified) |
| **CONSTANT_ALPHA** | 0.005 | Constant learning rate value |

### Parameter Adjustment

Experimental settings can be adjusted by modifying parameters in the `scripts/chapter5_experiment.sh` file. For example:
- **Learning Rate**: Adjust the learning rate for gradient descent algorithms.
- **Batch Size**: Set the number of samples per update in Mini-Batch Gradient Descent (MBGD).
- **Iterations**: Set the maximum number of iterations for each algorithm.

## Experimental Results

The experiment will generate **3 independent visualization results**, corresponding to the three gradient descent algorithms (BGD, SGD, MBGD). Each algorithm's result contains **2 subplots** that respectively show the optimization path and error convergence process for intuitive comparison.

### Visualization Content

1.  **Optimization Path Trajectory Plot**
    -   **Shows**: Algorithm's optimization path in 2D space
    -   **Comparison Dimensions**:
        -   **Constant Learning Rate**: Optimization trajectory with fixed learning rate
        -   **Dynamic Learning Rate**: Optimization trajectory with learning rate that changes (e.g., decays with iterations)
    -   **Purpose**: Visually demonstrates how the algorithm gradually approaches the optimal solution from the initial point in 2D space, and how different learning rates affect search efficiency.

2.  **Error Convergence Plot**
    -   **Shows**: The descent curve of the objective function error (loss) as iterations increase
    -   **Comparison Dimensions**:
        -   **Constant Learning Rate**: Error descent trend with fixed learning rate
        -   **Dynamic Learning Rate**: Error descent trend with changing learning rate (e.g., decay with iterations)
    -   **Purpose**: Quantitative analysis of algorithm convergence speed, stability, and final accuracy, evaluating the effect of different learning rate strategies on error convergence.

### BGD Trajectory and Error Convergence
![BGD Trajectory and Error Convergence](./results/BGD_results.png)

### SGD Trajectory and Error Convergence
![SGD Trajectory and Error Convergence](./results/SGD_results.png)

### MBGD Trajectory and Error Convergence
![MBGD Trajectory and Error Convergence](./results/MBGD_results.png)
