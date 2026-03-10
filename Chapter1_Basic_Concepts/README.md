##  介绍

本章节实现了一个基础的网格世界（Grid World）强化学习环境。网格世界是强化学习中经典的基准测试环境，用于理解和测试各种强化学习算法。可配置的网格世界环境模型，并提供了可视化的功能。

## 文件结构

```bash
Chapter1_Basic_Concepts/
├── results/ # 实验结果
│ ├── grid_world_policy.png # 策略迭代可视化结果
│ └── grid_world_policy_value.png # 值迭代可视化结果
├── src/ # 源代码
│ ├── environment_model.py # 网格世界环境模型
│ ├── experiment.py # 实验主文件
│ └── visualization.py # 可视化工具
└── scripts/ # 运行脚本目录
└── chapter1_experiment.sh # 一键运行实验脚本
```

##  快速开始

```bash
bash Chapter1_Basic_Concepts/scripts/chapter1_experiment.sh
```

## 实验结果
实验将生成两个可视化图表，展示随机策略及其对应的状态值：

### 随机策略可视化
![随机策略图](./results/grid_world_policy.png)


### 随机策略下的状态值函数
![策略价值图](./results/grid_world_policy_value.png)
