# Foundations-of-RL-Learning-From-Scratch

> 🎯 从零开始构建强化学习基础算法学习库 | 🧠 理解经典RL算法的核心原理与实践

[![Algorithm: Reinforcement Learning Basics](https://img.shields.io/badge/Algorithm-Reinforcement%20Learning%20Basics-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status: Updating](https://img.shields.io/badge/Status-Updating-yellow.svg)]()

[English](README_en.md) | [简体中文](README.md)

---

## 📋 目录
- [🌟 项目简介](#-项目简介)
- [✨ 项目亮点](#-项目亮点)
- [🚀 快速开始](#-快速开始)
- [🏗️ 项目结构与算法概览](#-项目结构与算法概览)
- [📖 核心学习资源](#-核心学习资源)
- [📄 许可证](#-许可证)

## 🌟 项目简介

**Foundations-of-RL-Learning-From-Scratch** 是一个教学性质的、从零实现的强化学习基础算法库。本项目的目标是：**通过简洁、清晰、可运行的代码，帮助学习者直观地理解强化学习核心算法的原理、推导与实现细节**。

项目从基础的网格世界环境、贝尔曼方程开始，逐步实现策略迭代、价值迭代、蒙特卡洛、时序差分、价值函数近似等经典算法，直至策略梯度算法。每个算法模块都包含：
- 📖 **完整的代码实现** 与 **详尽的注释**
- 💻 **可执行的测试示例**
- 🛠️**支持调整关键参数（如学习率、折扣因子、探索率等）**
- 📊 **学习过程与结果的可视化**

## ✨ 项目亮点
- **代码驱动学习**：拒绝“黑箱”，每个算法都有手把手实现的代码，可逐行调试。
- **统一的网格世界环境**：所有算法在同一个简单的网格环境中测试，便于对比和理解。
- **结构清晰，循序渐进**：章节安排符合经典教材的学习路径，从基础到进阶。
- **理论与实践结合**：代码实现紧密配合《强化学习的数学原理》等权威教材。

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/tequila28/Foundations-of-RL-Learning.git
```

### 2. 创建环境并安装依赖
推荐使用 `conda` 或 `venv` 管理环境。

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

## 🏗️ 项目结构

## 📊 算法实现概览

项目按核心概念和算法类别组织为不同章节：

| 算法名称 | 状态 | 所在章节 | 核心说明 |
| :--- | :---: | :--- | :--- |
| **网格世界环境** | ✅ | `Chapter1_Basic_Concepts/` | 用于测试所有算法的统一仿真环境。 |
| **贝尔曼方程** | ✅ | `Chapter2_Bellman_Equations/` | 强化学习的核心递归方程，价值计算的基础。 |
| **策略迭代 (Policy Iteration)** | ✅ | `Chapter3_Policy_and_Value_Iteration/` | 经典DP算法：策略评估 + 策略改进的交替迭代。 |
| **价值迭代 (Value Iteration)** | ✅ | `Chapter3_Policy_and_Value_Iteration/` | 更高效的DP方法，直接迭代价值函数至最优。 |
| **蒙特卡洛方法 (MC)** | ✅ | `Chapter4_Monte_Carlo/` | 基于完整回合（episode）采样的免模型预测与控制。 |
| **随机近似与优化** | ✅ | `Chapter5_Stochastic_Approximation/` | 实现SGD、BGD等优化器，为后续算法打下基础。 |
| **时序差分学习 (TD Learning)** | ✅ | `Chapter6_Temporal_Difference/` | 包括TD(0)、SARSA、Q-learning等核心算法。 |
| **值函数近似 (Value Function Approximation)** | ✅ | `Chapter7_Function_Approximation/` | 使用线性函数或神经网络近似大规模/连续状态的价值函数。 |
| **策略梯度方法 (Policy Gradient)** | 🚧 规划中 | `(待补充)` | REINFORCE、Actor-Critic 等直接优化策略的算法。 |

> 每个章节的目录下通常包含：算法实现文件（`.py`）、代码运行脚本（`.sh`、必要的说明文档（`README.md`）以及可视化结果。



## 📖 相关资源

本项目的学习与实现深受以下优质开源资源的启发，强烈推荐与代码结合学习：

1.  **核心教材**：
    🔗 **[《强化学习的数学原理》(Mathematical Foundations of Reinforcement Learning)](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)** - 由西湖大学赵世钰老师撰写。本书从数学角度系统讲解RL，逻辑严谨，以网格世界贯穿始终，是理解算法背后原理的绝佳指南。该GitHub仓库包含书籍PDF、讲义幻灯片等全部资源。

2.  **配套视频课程**：
    🔗 **[【强化学习的数学原理】课程](https://www.bilibili.com/video/BV1sd4y167NS)** - 与教材配套的完整视频课程（Bilibili平台）。通过视频讲解，可以更直观地理解复杂概念。


## 📄 许可证

本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。
