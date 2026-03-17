# Chapter 5: Stochastic Approximation

## 实验介绍

本章节主要研究随机近似（Stochastic Approximation）方法在强化学习中的应用。随机近似是一种用于优化的技术，特别适用于处理具有随机性或不确定性的环境。通过实现和比较不同的梯度下降算法（BGD、SGD、MBGD），我们可以深入理解这些方法的收敛特性和性能差异。

## 文件结构

```
Chapter5_Stochastic_Approximation/
├── results/                    # 主实验结果存储目录
├── scripts/                    # 实验脚本
│   └── chapter5_experiment.sh  # 用于运行实验的脚本
├── src/                        # 源代码目录
│   ├── environment.py         # 环境定义
│   ├── experiment.py         # 实验逻辑实现
│   ├── visualization.py      # 可视化工具
│   ├── algorithms/           # 算法实现目录
│   │   ├── __init__.py
│   │   ├── bgd.py           # 批量梯度下降算法
│   │   ├── mbgd.py          # 小批量梯度下降算法
│   │   ├── sgd.py           # 随机梯度下降算法
│   │   └── __pycache__/     # Python缓存文件（通常.gitignore忽略）
│   └── data/                 # 数据目录（建议添加）
└── README.md                 # 项目说明文档
```

## 快速开始

### 运行命令

运行实验脚本：
   ```bash
   Chapter5_Stochastic_Approximation/scripts/chapter5_experiment.sh
   ```
   该脚本将运行所有实验并将结果保存到 `results/` 目录中。

