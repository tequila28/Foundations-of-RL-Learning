"""Regenerate the eight chapter animations embedded in their READMEs."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.gridworld_lab.chapter_defaults import animation_config, script_path
from apps.gridworld_lab.generate_previews import PREVIEW_CASES
from apps.gridworld_lab.runner import run
from apps.gridworld_lab.visualize import save_estimation_gif, save_exploration_gif

GIF_TITLES = {
    "Chapter1_Basic_Concepts": "Random policy",
    "Chapter2_Bellman_Equations": "Iterative policy evaluation",
    "Chapter3_Policy_and_Value_Iteration": "Value iteration",
    "Chapter4_Monte_Carlo": "Monte Carlo epsilon-greedy",
    "Chapter5_Stochastic_Approximation": "SGD",
    "Chapter6_Temporal_Difference": "SARSA",
    "Chapter7_Value_Function_Approximation": "Linear TD(0)",
    "Chapter8_Policy_Gradient": "REINFORCE",
}


def main():
    for chapter, algorithm, directory in PREVIEW_CASES:
        config = animation_config(directory)
        print(f"Running {directory} with {script_path(directory).name} defaults: "
              f"episodes={config['episodes']}, max_steps={config['max_steps']}", flush=True)
        result = run(chapter, algorithm, config)
        output = ROOT / directory / "results" / "interactive_lab_preview.gif"
        if result["kind"] == "estimation":
            save_estimation_gif(result, output, title=GIF_TITLES[directory])
        else:
            save_exploration_gif(result, output, title=GIF_TITLES[directory])
        print(f"Saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
