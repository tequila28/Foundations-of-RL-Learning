
## 时序差分（TD）算法实验

本章节实现了五种时序差分（Temporal Difference, TD）算法，并在网格世界（Grid World）环境中可视化每种方法最终学到的策略及状态值。支持可配置的环境模型，提供多算法对比、训练曲线等可视化功能。

## 文件结构

```bash
Chapter6_Temporal_Difference/
├── results/
│   ├── final_policies_comparison.png
│   └── multi_algorithm_comparison.png
├── scripts/
│   └── chapter6_experiment.sh
└── src/
    ├── algorithms/
    │   ├── expected_sarsa.py
    │   ├── n_step_sarsa.py
    │   ├── off_policy_qlearning.py
    │   ├── on_policy_qlearning.py
    │   └── sarsa.py
    ├── experiment.py
    └── visualization.py
```

## 快速开始


```bash
bash Chapter6_Temporal_Difference/scripts/chapter6_experiment.sh
```


## 算法实现

### 1. SARSA
- 标准时序差分SARSA算法
- ε-greedy探索策略
- 在线更新Q值

### 2. Expected SARSA
- 期望式SARSA算法
- 以期望动作值更新
- 更平滑的策略改进

### 3. N-step SARSA
- 多步时序差分SARSA
- n步回报，兼顾短期与长期
- 可调节步数n

### 4. On-policy Q-Learning
- 在线Q-Learning算法
- ε-greedy行为策略
- 即时更新Q表

### 5. Off-policy Q-Learning
- 离策略Q-Learning算法
- 行为策略与目标策略分离
- 学习最优策略

## 实验结果

实验将生成多种可视化结果，展示各TD算法的学习效果与训练过程。

### 多算法最佳策略及状态值对比
![TD算法状态值对比](./results/final_policies_comparison.png)

### 多算法训练曲线对比
![TD算法训练曲线对比](./results/multi_algorithm_comparison.png)

---
如需自定义参数，建议修改脚本文件。
