"""Read the parameter assignments in each chapter's existing Bash script."""

from pathlib import Path
import re
import shlex

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {
    "Chapter1_Basic_Concepts": "chapter1_experiment.sh",
    "Chapter2_Bellman_Equations": "chapter2_experiment.sh",
    "Chapter3_Policy_and_Value_Iteration": "chapter3_experiment.sh",
    "Chapter4_Monte_Carlo": "chapter4_experiment.sh",
    "Chapter5_Stochastic_Approximation": "chapter5_experiment.sh",
    "Chapter6_Temporal_Difference": "chapter6_experiment.sh",
    "Chapter7_Value_Function_Approximation": "chapter7_experiment_one.sh",
    "Chapter8_Policy_Gradient": "chapter8_experiment.sh",
}


def script_path(chapter_directory):
    return ROOT / chapter_directory / "scripts" / SCRIPTS[chapter_directory]


def read_assignments(chapter_directory):
    """Read literal NAME=value defaults; never execute the shell script."""
    assignments = {}
    for line in script_path(chapter_directory).read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]*)=(.*)$", line.strip())
        if match:
            words = shlex.split(match.group(2), comments=True)
            if words:
                assignments[match.group(1)] = " ".join(words)
    return assignments


def _numbers(value):
    return [int(item) for item in value.split()]


def animation_config(chapter_directory):
    """Map the selected experiment's Bash defaults onto the shared runner."""
    values = read_assignments(chapter_directory)
    config = {
        "seed": int(values.get("SEED", 42)),
        "size": int(values.get("SIZE", 5)),
        "gamma": float(values.get("GAMMA", .9)),
        "target": _numbers(values.get("TARGET_STATES", "17"))[0],
        "start": 0,  # Scripts without a starting state need a replay starting point.
        "forbidden": _numbers(values.get("FORBIDDEN_STATES", "6 7 12 16 18 21")),
        "r_step": float(values.get("R_DEFAULT", 0)),
        "r_forbid": float(values.get("R_FORBID", -1)),
        "r_target": float(values.get("R_TARGET", 1)),
        "r_wall": float(values.get("R_BOUND", -1)),
        "episodes": 80,
        "max_steps": 25,  # Replay length when the Bash script does not define one.
        "alpha": .1,
        "epsilon": .1,
        "iterations": 500,
        "theta": 1e-5,
        "inner_iterations": 10,
        "samples": 400,
        "batch_size": 32,
        "hidden_size": 32,
    }
    if chapter_directory.startswith("Chapter3_"):
        config.update(iterations=int(values["MAX_ITERATION"]), theta=float(values["THETA"]),
                      inner_iterations=int(values["TRUNCATED_INNER_ITERATION"]))
    elif chapter_directory.startswith("Chapter4_"):
        # The GIF shows the first epsilon-greedy experiment from this comparison script.
        config.update(episodes=int(values["MC_EPS1_ITERATIONS"]),
                      max_steps=int(values["MC_EPS1_EPISODE_LENGTH"]),
                      epsilon=float(values["MC_EPS1_EPSILON"]))
    elif chapter_directory.startswith("Chapter5_"):
        config.update(samples=int(values["NUM_SAMPLES"]),
                      episodes=int(values["TOTAL_ITERATIONS"]),
                      square_size=float(values["SQUARE_SIZE"]),
                      init_x=float(values["INIT_X"]), init_y=float(values["INIT_Y"]),
                      batch_size=_numbers(values["BATCH_SIZES"])[0],
                      alpha=float(values["CONSTANT_ALPHA"]),
                      use_original_sampling=True)
    elif chapter_directory.startswith("Chapter6_"):
        config.update(episodes=int(values["NUM_EPISODES"]), max_steps=int(values["MAX_STEPS"]),
                      alpha=float(values["LEARNING_RATE"]), epsilon=float(values["EPSILON"]),
                      epsilon_decay=float(values["EPSILON_DECAY"]),
                      epsilon_min=float(values["EPSILON_MIN"]))
    elif chapter_directory.startswith("Chapter7_"):
        # experiment_one/experiment.py generates episodes with max_steps=500.
        config.update(episodes=int(values["N_EPISODES"]), max_steps=500,
                      alpha=float(values["LEARNING_RATE"]))
    elif chapter_directory.startswith("Chapter8_"):
        config.update(episodes=int(values["NUM_EPISODES"]), max_steps=int(values["MAX_STEPS"]),
                      hidden_size=int(values["HIDDEN_SIZE"]), alpha=float(values["LEARNING_RATE"]))
    return config
