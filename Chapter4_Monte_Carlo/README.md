##  蒙特卡洛算法实验

本章节实现了求解蒙特卡洛算法三种算法等等并在网格世界（Grid World）中可视化每种方法最后学到的策略及其状态值。包括可配置的网格世界环境模型，并提供了可视化的功能。同时针对不同eplison值可视化mc epsilon greedy算法的散点图

## 文件结构

```bash
Chapter4_Monte_Carlo/
├── results/
│   ├── mc_policy_comparison.png
│   └── state_action_visit_scatter.png
├── scripts/
│   └── chapter4_experiment.sh
└── src/
    ├── algorithms/
    │   ├── mc_basic.py
    │   ├── mc_epsilon_greedy.py
    │   └── mc_exploring_starts.py
    ├── experiment.py
    └── visualization.py
```

##  快速开始

```bash
bash Chapter4_Monte_Carlo/scripts/chapter4_experiment.sh
```

## 算法实现

### 1. MC Basic (基本蒙特卡洛控制)
- 基于每次访问的蒙特卡洛更新
- 无ε-greedy探索机制
- 固定策略评估

### 2. MC Exploring Starts (探索起点蒙特卡洛)
- 每个episode从随机的状态-动作对开始
- 保证对状态-动作空间的充分探索
- 采用贪婪策略改进

### 3. MC ε-greedy (ε-贪心蒙特卡洛)
- 平衡探索与利用的ε-greedy策略
- 可调节的探索率ε
- 增量式策略改进

## 实验结果

实验将生成2个可视化结果，分别展示三种算法的学习效果和探索特性。


### 三种算法最佳策略及其状态值对比
![蒙特卡洛算法状态值对比](./results/mc_policy_comparison.png)

### 不同ε值下状态-动作对访问频率散点图
- 展示ε=1.0（完全探索）和ε=0.2（部分探索）下的探索行为差异
- 横轴：状态-动作对索引
- 纵轴：访问次数
![状态-动作对访问频率散点图](./results/state_action_visit_scatter.png)
