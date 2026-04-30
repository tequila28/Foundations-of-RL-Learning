## 介绍

本章节实现了一个基础的网格世界（Grid World）强化学习环境。网格世界是强化学习中经典的基准测试环境，用于理解和测试各种强化学习算法。本项目实现了可配置的网格世界环境模型，并提供了策略可视化与状态值函数可视化的功能，为后续算法实验提供统一的基准环境。

---

<details>
<summary><strong>English / 中文</strong> (点击切换语言 | Click to switch language)</summary>

## Introduction

This chapter implements a basic Grid World reinforcement learning environment. Grid World is a classic benchmark environment in reinforcement learning, used for understanding and testing various reinforcement learning algorithms. This project implements a configurable Grid World environment model and provides functionalities for policy visualization and state-value function visualization, offering a unified benchmark environment for subsequent algorithm experiments.

</details>

---

## 文件结构 | File Structure

```bash
Chapter1_Basic_Concepts/
├── results/ # 实验结果 | Experiment results
│ ├── grid_world_policy.png # 策略迭代可视化结果 | Policy iteration visualization
│ └── grid_world_policy_value.png # 值迭代可视化结果 | Value iteration visualization
├── src/ # 源代码 | Source code
│ ├── environment_model.py # 网格世界环境模型 | Grid World environment model
│ ├── experiment.py # 实验主文件 | Main experiment file
│ └── visualization.py # 可视化工具 | Visualization utilities
└── scripts/ # 运行脚本目录 | Scripts directory
└── chapter1_experiment.sh # 一键运行实验脚本 | One-click experiment script
```

##  快速开始

```bash
bash Chapter1_Basic_Concepts/scripts/chapter1_experiment.sh
```
## 参数说明

以下是实验脚本中使用的主要参数及其含义：

| 参数 | 值 | 说明 |
|------|-----|------|
| **SIZE** | 5 | 方形网格世界的维度（行数和列数），这里创建 5×5 的网格 |
| **GAMMA** | 0.9 | 强化学习中的未来奖励折扣因子，取值 0-1，值越接近 1 表示越重视未来奖励 |
| **ACTIONS** | "up right down left stay" | 智能体可采取的动作集合，"stay"表示停留在当前格子 |
| **FORBIDDEN_STATES** | "6 7 12 16 18 21" | 进入这些状态会收到惩罚 |
| **TARGET_STATES** | "17" | 目标状态索引列表，到达这些状态会收到 奖励 |
| **R_BOUND** | -1 | 当智能体动作会使其移出网格边界（撞墙）时获得的即时奖励，这是对无效移动的惩罚 |
| **R_FORBID** | -1 | 当智能体进入禁止状态时获得的即时奖励，这是对进入危险区域的惩罚 |
| **R_TARGET** | 1 | 当智能体到达目标状态时获得的即时奖励，这是达成目标的正向奖励 |
| **R_DEFAULT** | 0 | 任何其他有效转移（移动到非目标、非禁止格子）时获得的默认即时奖励 |
| **SEED** | 42 | 用于初始化伪随机数生成器的随机种子值，确保实验结果的复现性 |

## 实验结果
实验将生成两个可视化图表，展示随机策略及其对应的状态值：

### 随机策略可视化
![随机策略图](./results/grid_world_policy.png)


### 随机策略下的状态值
![策略价值图](./results/grid_world_policy_value.png)
