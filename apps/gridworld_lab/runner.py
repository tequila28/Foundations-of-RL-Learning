"""Chapter selection and dispatch for the interactive learning lab."""

import numpy as np

from apps.gridworld_lab import common
from apps.gridworld_lab.chapters import chapter1, chapter2, chapter3, chapter4, chapter5, chapter6, chapter7, chapter8


CHAPTERS = {
    "第 1 章 · 基础概念": ["随机策略与网格环境"],
    "第 2 章 · Bellman 方程": ["策略评估（迭代）", "策略评估（闭式解）"],
    "第 3 章 · 动态规划": ["价值迭代", "策略迭代", "截断策略迭代"],
    "第 4 章 · 蒙特卡洛": ["MC ε-greedy", "MC Exploring Starts"],
    "第 5 章 · 随机近似": ["SGD", "Mini-batch GD", "Batch GD"],
    "第 6 章 · 时序差分": ["SARSA", "Expected SARSA", "Q-learning"],
    "第 7 章 · 函数近似": ["线性 TD(0)"],
    "第 8 章 · 策略梯度": ["REINFORCE", "Actor-Critic"],
}


def run(chapter_name, algorithm, config):
    """Run one selected experiment and add the environment for plotting."""
    config = {**config, "algorithm": algorithm}
    if chapter_name.startswith("第 5 章"):
        return chapter5.run(config)

    env = common.make_environment(config)
    rng = np.random.default_rng(int(config["seed"]))
    modules = {
        "第 1 章": chapter1,
        "第 2 章": chapter2,
        "第 3 章": chapter3,
        "第 4 章": chapter4,
        "第 6 章": chapter6,
        "第 7 章": chapter7,
    }
    if chapter_name.startswith("第 8 章"):
        result = chapter8.run(env, config)
    else:
        prefix = next((key for key in modules if chapter_name.startswith(key)), None)
        if prefix is None:
            raise ValueError(f"Unknown chapter: {chapter_name}")
        result = modules[prefix].run(env, config, rng)
    return {"kind": "gridworld", "env": env, **result}
