# Foundations-of-RL-Learning-From-Scratch

> 🎯 Building Reinforcement Learning Foundation Algorithms from Scratch | 🧠 Understanding the Core Principles and Practices of Classical RL Algorithms

[![Algorithm: Reinforcement Learning Basics](https://img.shields.io/badge/Algorithm-Reinforcement%20Learning%20Basics-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status: Updating](https://img.shields.io/badge/Status-Updating-yellow.svg)]()

[English](README_en.md) | [简体中文](README.md)

---

## 📋 Table of Contents
- [🌟 Project Introduction](#-project-introduction)
- [✨ Project Highlights](#-project-highlights)
- [🚀 Quick Start](#-quick-start)
- [🏗️ Project Structure & Algorithm Overview](#-project-structure--algorithm-overview)
- [📖 Key Learning Resources](#-key-learning-resources)
- [📄 License](#-license)

## 🌟 Project Introduction

**Foundations-of-RL-Learning-From-Scratch** is an educational, from-scratch implementation library of foundational reinforcement learning algorithms. The goal of this project is: **to help learners intuitively understand the principles, derivations, and implementation details of core RL algorithms through concise, clear, and runnable code**.

Starting from the basic grid world environment and Bellman equations, the project gradually implements classic algorithms such as policy iteration, value iteration, Monte Carlo methods, temporal-difference learning, value function approximation, up to policy gradient methods. Each algorithm module includes:
- 📖 **Complete code implementation** with **detailed annotations**
- 💻 **Executable test examples**
- 🛠️ **Support for adjusting key parameters** (e.g., learning rate, discount factor, exploration rate)
- 📊 **Visualization of the learning process and results**

## ✨ Project Highlights
- **Code-Driven Learning**: Reject the "black box" approach; each algorithm is implemented step-by-step with code that can be debugged line by line.
- **Unified Grid World Environment**: All algorithms are tested in the same simple grid environment, facilitating comparison and understanding.
- **Clear Structure, Progressive Learning**: Chapter organization follows the learning path of classic textbooks, from basics to advanced topics.
- **Theory Meets Practice**: Code implementations are closely aligned with authoritative textbooks like "Mathematical Foundations of Reinforcement Learning".

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/tequila28/Foundations-of-RL-Learning.git
```

### 2. Create Environment and Install Dependencies
It is recommended to use `conda` or `venv` for environment management.

```bash
conda create -n rl_learning python=3.9 -y
conda activate rl_learning
```
```bash
python -m venv rl_learning
source rl_learning/bin/activate # Linux/Mac
```
```bash
pip install numpy matplotlib
```

## 🏗️ Project Structure & Algorithm Overview

The project is organized into different chapters based on core concepts and algorithm categories:

| Algorithm | Status | Location | Core Description |
| :--- | :---: | :--- | :--- |
| **Grid World Environment** | ✅ | `Chapter1_Basic_Concepts/` | Unified simulation environment for testing all algorithms. |
| **Bellman Equations** | ✅ | `Chapter2_Bellman_Equations/` | The core recursive equations of RL, foundational for value computation. |
| **Policy Iteration** | ✅ | `Chapter3_Policy_and_Value_Iteration/` | Classic DP algorithm: alternating iterations of policy evaluation and policy improvement. |
| **Value Iteration** | ✅ | `Chapter3_Policy_and_Value_Iteration/` | A more efficient DP method that directly iterates the value function to optimality. |
| **Monte Carlo Methods (MC)** | ✅ | `Chapter4_Monte_Carlo/` | Model-free prediction and control based on complete episode sampling. |
| **Stochastic Approximation & Optimization** | ✅ | `Chapter5_Stochastic_Approximation/` | Implements optimizers like SGD, BGD, laying the groundwork for subsequent algorithms. |
| **Temporal-Difference Learning (TD Learning)** | ✅ | `Chapter6_Temporal_Difference/` | Includes core algorithms like TD(0), SARSA, and Q-learning. |
| **Value Function Approximation** | ✅ | `Chapter7_Function_Approximation/` | Approximates value functions for large-scale/continuous states using linear functions or neural networks. |
| **Policy Gradient Methods** | 🚧 Planned | `(To be added)` | Algorithms that directly optimize the policy, such as REINFORCE and Actor-Critic. |

> Each chapter directory typically contains: algorithm implementation files (`.py`), demo/execution scripts, necessary documentation (`README.md`), and visualization results.

## 📖 Key Learning Resources

The learning and implementation of this project are greatly inspired by the following high-quality open-source resources. They are highly recommended for combined study with the code:

1.  **Core Textbook**:
    🔗 **[《Mathematical Foundations of Reinforcement Learning》](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)** - Authored by Prof. Shiyu Zhao from Westlake University. This book systematically explains RL from a mathematical perspective. It is logically rigorous, uses a grid world throughout, and is an excellent guide for understanding the principles behind the algorithms. The GitHub repository contains all resources including the book PDF and lecture slides.

2.  **Complementary Video Course**:
    🔗 **[【Mathematical Foundations of Reinforcement Learning】Course](https://www.bilibili.com/video/BV1sd4y167NS)** - A complete video course (on Bilibili platform) complementing the textbook. Video explanations help visualize complex concepts.

## 📄 License

This project is open source under the MIT License. See the [LICENSE](LICENSE) file for details.
