"""Regenerate the chapter preview figures embedded in the chapter READMEs."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.gridworld_lab.runner import run
from apps.gridworld_lab.visualize import estimation_figures, grid_figure


PREVIEW_CASES = [
    ("第 1 章 · 基础概念", "随机策略与网格环境", "Chapter1_Basic_Concepts"),
    ("第 2 章 · Bellman 方程", "策略评估（迭代）", "Chapter2_Bellman_Equations"),
    ("第 3 章 · 动态规划", "价值迭代", "Chapter3_Policy_and_Value_Iteration"),
    ("第 4 章 · 蒙特卡洛", "MC ε-greedy", "Chapter4_Monte_Carlo"),
    ("第 5 章 · 随机近似", "SGD", "Chapter5_Stochastic_Approximation"),
    ("第 6 章 · 时序差分", "SARSA", "Chapter6_Temporal_Difference"),
    ("第 7 章 · 函数近似", "线性 TD(0)", "Chapter7_Value_Function_Approximation"),
    ("第 8 章 · 策略梯度", "REINFORCE", "Chapter8_Policy_Gradient"),
]


def main():
    base_config = {
        "seed": 42, "size": 5, "gamma": 0.9, "target": 17, "start": 0,
        "forbidden": [6, 7, 12, 16, 18, 21], "r_step": -0.04,
        "r_forbid": -1.0, "r_target": 1.0, "r_wall": -0.04,
        "episodes": 80, "max_steps": 25, "alpha": 0.1, "epsilon": 0.1,
        "iterations": 500, "theta": 1e-5, "inner_iterations": 10,
        "samples": 400, "batch_size": 32, "hidden_size": 32,
    }
    for chapter, algorithm, directory in PREVIEW_CASES:
        result = run(chapter, algorithm, base_config)
        if result["kind"] == "estimation":
            figure = estimation_figures(result)
        else:
            figure = grid_figure(result)
        target = ROOT / directory / "results" / "interactive_lab_preview.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=160, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
