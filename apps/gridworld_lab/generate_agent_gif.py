"""Generate the Q-learning exploration animation shown in the READMEs."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.gridworld_lab.runner import run
from apps.gridworld_lab.chapter_defaults import animation_config
from apps.gridworld_lab.visualize import save_exploration_gif


def main():
    config = animation_config("Chapter6_Temporal_Difference")
    result = run("第 6 章 · 时序差分", "Q-learning", config)
    target = ROOT / "apps" / "gridworld_lab" / "assets" / "gridworld_agent_exploration.gif"
    save_exploration_gif(result, target, fps=3, title="Q-learning ε-greedy exploration")
    print(f"Saved {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
