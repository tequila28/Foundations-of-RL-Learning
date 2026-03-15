# Foundations-of-RL-Learning-From-Scratch

> 🎯 从零开始构建强化学习基础算法学习库 | 🧠 理解经典RL算法的核心原理与实践

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/) 
[![Status](https://img.shields.io/badge/Status-updating-yellow.svg)]() 
[![Algorithm](https://img.shields.io/badge/Algorithm-Reinforcement%20Learning%20Basics-blue.svg)]()

## 项目简介

**Foundations-of-RL-Learning-From-Scratch** 是一个教学性质的强化学习基础算法实现库，旨在通过简洁、清晰的代码帮助学习者深入理解强化学习算法的核心原理。本项目从基础的贝尔曼方程开始，逐步扩展到蒙特卡洛方法、时序差分学习、函数近似和策略梯度等现代强化学习方法。

每个算法都有完整的实现、详细的代码注释和对应的实验示例，让学习者能够：
- 📖 理解算法背后的数学原理
- 💻 亲手实现核心算法逻辑
- 🔬 通过实验验证算法性能
- 📊 可视化学习过程和结果

## 📦 下载与安装依赖

### 安装步骤

**克隆项目**
```bash
git clone https://github.com/tequila28/Foundations-of-RL-Learning.git
```
**创建虚拟环境**

```bash
conda create -n rl_learning python=3.9 -y
conda activate rl_learning
```
**安装依赖**
```bash
pip install numpy matplotlib
```
## 🏗️ 项目结构

## 📊 算法实现概览

| 算法名称             | 实现状态 | 位置                     | 说明                                                                 |
|----------------------|----------|--------------------------|----------------------------------------------------------------------|
| 网格环境             | ✅ 已实现 | `Chapter1_Basic_Concepts` | 用于算法测试的网格世界仿真环境，支持状态、动作、奖励与终止条件。         |
| 贝尔曼方程           | ✅ 已实现 | `Chapter2_Bellman_Equations` | 强化学习核心方程，用于递归计算状态价值或动作价值函数。                 |
| 策略迭代 (Policy Iteration) | ✅ 已实现 | `Chapter3_Policy_and_Value_Iteration` | 经典动态规划算法，通过策略评估+策略改进交替迭代，收敛至最优策略。       |
| 价值迭代 (Value Iteration) | ✅ 已实现 | `Chapter3_Policy_and_Value_Iteration` | 动态规划方法，直接迭代更新价值函数至收敛，再导出最优策略，效率更高。     |
| 蒙特卡洛 (First-Visit MC) | ✅ 已实现 | `Chapter4_Monte_Carlo` | 基于完整回合采样（Episode）的免模型学习方法，用于估计状态/策略价值。     |
| 随机近似 | ❌ 未实现 | 待补充 | 包含随机梯度下降（SGD）、批量梯度下降（BGD）及小批量梯度下降（MBGD）等优化算法。 |
| 时序差分算法 (TD) | ❌ 未实现 | 待补充 | 结合了蒙特卡洛采样和动态规划自举（Bootstrapping）思想的一类算法，包括TD、SARSA及其变体、Q-learning等。 |
| 值函数近似算法 | ❌ 未实现 | 待补充 | 使用参数化函数（如线性函数、神经网络）来近似表示大规模或连续状态空间下的价值函数。 |
| 策略梯度算法 | ❌ 未实现 | 待补充 | 直接对参数化策略进行优化的一类算法，通过梯度上升来最大化期望回报，主要包括REINFORCE和Actor-Critic（AC）等。 |



## 📖 相关资源

本项目的学习与实现参考了丰富的开源资料，主要包括一本备受好评的教材和一个完整的配套视频课程。

### 核心教材

**[《强化学习的数学原理》 (Mathematical Foundations of Reinforcement Learning)》](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)**

### 配套视频课程

**[【强化学习的数学原理】课程：从零开始到透彻理解（完结）](https://www.bilibili.com/video/BV1sd4y167NS)**

该书由西湖大学赵世钰老师撰写，是系统学习强化学习基础理论的权威指南。其核心特色是数学严谨、循序渐进，全书以网格世界为统一实例，从MDP、贝尔曼方程等基础概念出发，逐步推导至策略梯度、Actor-Critic等核心算法。随书还附有完整的教材PDF、讲义及配套代码，非常适合理论与实践相结合的学习。

## 🤝 致谢

