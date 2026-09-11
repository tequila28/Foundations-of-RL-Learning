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
- [📚 Knowledge Summary Documents](#-knowledge-summary-documents)
- [🚀 Quick Start](#-quick-start)
- [🧩 Environment and Dependencies](#-environment-and-dependencies)
- [🏗️ Project Structure & Algorithm Overview](#-project-structure--algorithm-overview)
- [📖 Key Learning Resources](#-key-learning-resources)
- [📄 License](#-license)

## 🌟 Project Introduction

### 📈 Why Learn Reinforcement Learning?

In recent years, Reinforcement Learning (RL) has become a shining star in the field of Large Language Models (LLM), from ChatGPT's RLHF (Reinforcement Learning from Human Feedback) o1 series to DeepSeek's R1 series. RL has become a key technology for improving the alignment, safety, and reasoning capabilities of large language models. However, to deeply understand how modern RL trains LLMs, the best approach is to start from the beginning and learn the evolutionary path of classic RL algorithms.

**Foundations-of-RL-Learning-From-Scratch** is an educational, from-scratch implementation library of foundational reinforcement learning algorithms. The goal of this project is: **to help learners intuitively understand the principles, derivations, and implementation details of core RL algorithms through concise, clear, and runnable code**.

Starting from the basic grid world environment and Bellman equations, the project gradually implements classic algorithms such as policy iteration, value iteration, Monte Carlo methods, temporal-difference learning, value function approximation, up to policy gradient methods. Each algorithm module includes:
- 📖 **Complete code implementation** with **detailed annotations**
- 💻 **Executable test examples**
- 🛠️ **Support for adjusting key parameters** (e.g., learning rate, discount factor, exploration rate)
- 📊 **Visualization of the learning process and results**
- 📚 **Bilingual knowledge-summary PDF documents**: systematic summaries of each chapter's theory, code implementation, experiment setup, and result interpretation, helping readers build a traditional RL foundation for later LLM RL study.

## ✨ Project Highlights
- **Code-Driven Learning**: Reject the "black box" approach; each algorithm is implemented step-by-step with code that can be debugged line by line.
- **Unified Grid World Environment**: All algorithms are tested in the same simple grid environment, facilitating comparison and understanding.
- **Clear Structure, Progressive Learning**: Chapter organization follows the learning path of classic textbooks, from basics to advanced topics.
- **Theory Meets Practice**: Code implementations are closely aligned with authoritative textbooks like "Mathematical Foundations of Reinforcement Learning".
- **Bilingual Learning Notes**: Newly added Chinese and English PDF documents summarize each chapter around theory, implementation, experimental purpose, and result analysis.

## 📚 Knowledge Summary Documents

To help readers understand the project more completely, the repository now includes bilingual knowledge-summary PDF documents. Starting from the foundations of traditional reinforcement learning, these documents connect each chapter's code and experiments with core topics such as MDPs, Bellman equations, dynamic programming, Monte Carlo methods, TD learning, value function approximation, and policy gradients. They are also intended to provide a theoretical foundation for understanding reinforcement learning in LLM training.

| Language | Document | Description |
| :--- | :--- | :--- |
| Chinese | `docs/Foundations_of_RL_Learning_CN.pdf` | Complete Chinese guide to the theory and experiments. |
| English | `docs/Foundations_of_RL_Learning_EN.pdf` | English guide covering the same theory, implementation, and experiment analysis. |

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/tequila28/Foundations-of-RL-Learning.git
cd Foundations-of-RL-Learning
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

### 3. Install Python Dependencies

The project root now provides a `requirements.txt` file. The recommended installation command is:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

You can also install the dependencies manually:

```bash
pip install numpy matplotlib torch tqdm
```

> If you need the GPU version of PyTorch, please install the proper build for your CUDA version from the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## 🧩 Environment and Dependencies

Recommended environment:

| Item | Recommended Configuration |
| :--- | :--- |
| Python | 3.9 or later |
| NumPy | Numerical computation, matrix operations, and random sampling |
| Matplotlib | GridWorld plots, value functions, and training curves |
| tqdm | Progress bars during training |
| PyTorch | Policy-gradient and Actor-Critic neural network implementations in Chapter 8 |

Dependency file:

```bash
requirements.txt
```

It contains:

```text
numpy>=1.24
matplotlib>=3.7
tqdm>=4.66
torch>=2.0
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
| **Value Function Approximation** | ✅ | `Chapter7_Value_Function_Approximation/` | Approximates value functions for large-scale/continuous states using linear functions or neural networks. |
| **Policy Gradient Methods** | ✅ | `Chapter8_Policy_Gradient/`  | Algorithms that directly optimize the policy, such as REINFORCE and Actor-Critic. |

> Each chapter directory typically contains: algorithm implementation files (`.py`), demo/execution scripts, necessary documentation (`README.md`), and visualization results.

## 📖 Key Learning Resources

The learning and implementation of this project are greatly inspired by the following high-quality open-source resources. They are highly recommended for combined study with the code:

1.  **Core Textbook**:
    🔗 **[《Mathematical Foundations of Reinforcement Learning》](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)** - Authored by Prof. Shiyu Zhao from Westlake University. This book systematically explains RL from a mathematical perspective. It is logically rigorous, uses a grid world throughout, and is an excellent guide for understanding the principles behind the algorithms. The GitHub repository contains all resources including the book PDF and lecture slides.

2.  **Complementary Video Course**:
    🔗 **[【Mathematical Foundations of Reinforcement Learning】Course](https://www.bilibili.com/video/BV1sd4y167NS)** - A complete video course (on Bilibili platform) complementing the textbook. Video explanations help visualize complex concepts.

## 📄 License

This project is open source under the MIT License. See the [LICENSE](LICENSE) file for details.
