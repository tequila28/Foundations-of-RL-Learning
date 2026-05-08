# 章节8：策略梯度算法实验

<div align="right">

[English](README_en.md) | [简体中文](README.md)

</div>

## 介绍

### **策略梯度算法基础**

策略梯度算法是强化学习中一类直接优化策略的算法，通过参数化策略函数并沿着策略性能梯度的方向更新参数。与基于价值的方法不同，策略梯度方法直接学习最优策略，避免了值函数估计的偏差，特别适用于连续动作空间和高维状态空间问题。

### **REINFORCE算法**

REINFORCE（蒙特卡洛策略梯度）算法是基于完整轨迹回报的策略梯度方法。它通过采样完整轨迹计算累积回报，并使用这些回报作为权重来更新策略参数。算法通过引入基线函数（如状态值函数）来减少梯度估计的方差，提高学习稳定性。REINFORCE是策略梯度方法的基础，为后续Actor-Critic等算法提供了理论框架。

### **Actor-Critic算法**

Actor-Critic算法结合了策略梯度（Actor）和值函数估计（Critic）的优点。Actor负责选择动作，Critic负责评估状态价值，两者协同工作实现更高效的策略优化。该算法使用TD误差或优势函数作为策略更新的信号，实现单步或n步更新，避免了蒙特卡洛方法需要完整轨迹的限制，提高了样本利用率和学习效率。

### 算法实现

本章节在网格世界（Grid World）环境中实现了以下策略梯度算法组件以求解最佳策略：

1. **REINFORCE算法实现**
   - 基于蒙特卡洛策略梯度的无模型强化学习算法
   - 使用完整的episode回报来更新策略参数
   - 支持基线（baseline）减少方差，提高学习稳定性

2. **Actor-Critic算法实现**
   - 结合策略梯度（Actor）和值函数（Critic）的方法
   - 使用TD误差作为策略更新的信号
   - 支持优势函数估计，平衡探索与利用

3. **策略网络架构**
   - 支持多层感知机（MLP）架构
   - 可配置的隐藏层大小和激活函数
   - Softmax输出层用于生成动作概率分布

4. **价值网络架构**
   - 支持多层感知机（MLP）架构
   - 可配置的隐藏层大小
   - 输出单一标量值估计状态价值

5. **特征提取器设计**
   - **one-hot编码**：为每个状态生成唯一的热编码向量，提供稳定优化
   - **多项式特征**：扩展状态特征维度，但需注意收敛难度

## 文件结构

```bash
Chapter8_Policy_Gradient/
├── results/
│   ├── actor-critic.training_curves.png
│   ├── reinforce_training_curves.png
│   ├── reinforce_vs_actor-critic_episode_rewards.png
│   └── reinforce_vs_actor-critic_policy_comparison.png
├── scripts/
│   └── chapter8_experiment.sh
├── src/
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── reinforce_agent.py
│   │   ├── actor_critic_agent.py
│   │   ├── policy_network.py
│   │   └── feature_extractor.py
│   ├── experiment.py
│   └── visualization.py
└── README.md
```

## 快速开始

运行实验

```bash
bash Chapter8_Policy_Gradient/scripts/chapter8_experiment.sh
```

## 参数配置

以下是实验中使用的关键参数及其含义：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **GridWorld 环境配置** | | |
| **SIZE** | 5 | 网格世界的维度，创建 5×5 的方形网格 |
| **GAMMA** | 0.9 | 未来奖励的折扣因子 |
| **FORBIDDEN_STATES** | "6 7 12 16 18 21" | 禁止进入的状态列表 |
| **TARGET_STATES** | "17" | 目标/终止状态列表 |
| **R_BOUND** | -1 | 撞到网格边界时获得的即时奖励 |
| **R_FORBID** | -1 | 进入禁止状态时获得的即时奖励 |
| **R_TARGET** | 10 | 到达目标状态时获得的即时奖励 |
| **R_DEFAULT** | 0 | 正常状态转移时的默认即时奖励 |
| **NUM_EPISODES** | 5000 | 训练的总回合数 |
| **MAX_STEPS** | 20 | 每个回合的最大步数限制 |
| **算法超参数** | | |
| **HIDDEN_SIZE** | 128 | 神经网络隐藏层大小 |
| **LEARNING_RATE** | 0.001 | 学习率参数，控制每次更新的幅度 |
| **SEED** | 42 | 随机种子 |
| **FEATURE_TYPE** | "one_hot" | 特征提取类型，可选"one_hot"或"polynomial" |
| **Actor-Critic特定参数** | | |
| **GAE_LAMBDA** | 0.95 | 广义优势估计（GAE）的λ参数 |



## 实验结果

实验将生成四种可视化分析图表，全面展示REINFORCE和Actor-Critic两种策略梯度算法的学习效果与训练过程动态。

### 1. REINFORCE与Actor-Critic策略对比图
通过5×5网格世界中的策略箭头分布，直观对比两种算法学到的最终策略：
- **网格结构**：清晰显示5×5的网格布局
- **策略箭头**：每个单元格通过箭头方向表示学到的动作策略（上/下/左/右/停留）
- **特殊状态标记**：
  - 目标状态：蓝色显示
  - 禁止状态：橙色显示
- **对比分析**：并列展示两种算法的策略分布，便于直观比较策略差异
![策略梯度算法策略对比](./results/reinforce_vs_actor-critic_policy_comparison.png)

*该图展示两种算法在相同环境条件下学到的具体策略选择，便于分析策略收敛情况。*

### 2. 双算法训练回报对比图
通过曲线图对比两种算法的训练过程动态：
- **蓝色曲线**：REINFORCE算法回合回报收敛曲线
- **紫色曲线**：Actor-Critic算法回合回报收敛曲线
- **共同特征**：
  - 横轴：训练回合数
  - 纵轴：每回合累计回报
  - 包含移动平均线（平滑窗口）
  - 展示收敛速度和稳定性对比
![策略梯度算法回报对比](./results/reinforce_vs_actor-critic_episode_rewards.png)

*该图直观对比两种算法在训练过程中的回报收敛趋势和稳定性表现。*

### 3. REINFORCE算法训练过程分析
深入分析REINFORCE算法的训练动态，包含两个关键指标：
- **左部曲线**：回合累计回报变化趋势
  - 展示算法从探索到收敛的学习过程
  - 反映策略性能的逐步提升
- **右部曲线**：策略熵变化趋势
  - 量化策略探索程度
  - 熵值降低反映策略确定性增加
  - 帮助分析探索-利用平衡
![REINFORCE算法训练曲线](./results/reinforce_training_curves.png)

*该图从回报和探索两个维度全面分析REINFORCE算法的学习动态。*

### 4. Actor-Critic算法训练过程分析
深入分析Actor-Critic算法的训练动态，从三个关键维度全面评估算法性能：

- **左侧曲线：回合累计回报收敛过程**
  - 展示Actor-Critic算法特有的学习轨迹和收敛特性
  - 反映Critic网络对策略优化的加速效应

- **中部曲线：Actor损失函数变化**
  - 跟踪策略梯度损失（Actor loss）的演变趋势

- **右侧曲线：Critic损失函数变化**
  - 监测价值函数损失（Critic loss）的变化规律
  - 反映Critic网络对状态价值估计的准确性提升
  - 损失收敛表明Critic能提供稳定的TD误差信号

![Actor-Critic算法训练曲线](./results/actor-critic_training_curves.png)

*该图全面展示Actor-Critic算法在回报收敛和探索平衡方面的表现特征。*

