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
   bash Chapter5_Stochastic_Approximation/scripts/chapter5_experiment.sh
   ```
   该脚本将运行所有实验并将结果保存到 `results/` 目录中。

### 参数调整

可以通过修改 `scripts/chapter5_experiment.sh` 文件中的参数来调整实验设置。例如：
- **学习率（learning rate）**：调整梯度下降算法的学习率。
- **批量大小（batch size）**：在小批量梯度下降（MBGD）中设置每次更新的样本数量。
- **迭代次数（iterations）**：设置每种算法的最大迭代次数。


## 实验结果

实验将生成3个可视化结果，分别展示三种梯度下降算法的优化路径和误差收敛特性。

### 可视化内容

1. **轨迹图（Trajectories）**：展示不同梯度下降算法在参数空间中的优化路径。
2. **误差收敛图（Error Convergence）**：展示误差随迭代次数的变化趋势。

### BGD 轨迹与误差收敛
![BGD 轨迹与误差收敛](./results/BGD_results.png)

### SGD 轨迹与误差收敛
![SGD 轨迹与误差收敛](./results/SGD_results.png)

### MBGD 轨迹与误差收敛
![MBGD 轨迹与误差收敛](./results/MBGD_results.png)
