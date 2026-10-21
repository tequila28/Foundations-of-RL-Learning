"""Generate the Q-learning exploration animation shown in the READMEs."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.gridworld_lab.runner import run
from apps.gridworld_lab.visualize import save_exploration_gif


def main():
    config = {
        "seed": 21, "size": 5, "gamma": 0.9, "target": 17, "start": 0,
        "forbidden": [6, 7, 12, 16, 18, 21], "r_step": -0.04,
        "r_forbid": -1.0, "r_target": 1.0, "r_wall": -0.04,
        "episodes": 160, "max_steps": 32, "alpha": 0.18, "epsilon": 0.25,
    }
    result = run("第 6 章 · 时序差分", "Q-learning", config)
    target = ROOT / "apps" / "gridworld_lab" / "assets" / "gridworld_agent_exploration.gif"
    save_exploration_gif(result, target, fps=3)
    print(f"Saved {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
