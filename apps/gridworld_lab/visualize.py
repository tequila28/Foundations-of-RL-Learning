"""Visualizations for the learning lab, using the original GridWorld palette."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch, Rectangle


ARROWS = {"up": "↑", "right": "→", "down": "↓", "left": "←", "stay": "○"}
# Chapter1_Basic_Concepts/src/visualization.py defines these state colors.
GRID_COLORS = {"normal": "white", "forbidden": "orange", "target": "lightblue"}


def _grid_canvas(result):
    env, policy, values = result["env"], result["policy"], result["values"]
    size = env.size
    fig, ax = plt.subplots(figsize=(7, 6))
    for state in range(env.n_states):
        row, col = divmod(state, size)
        kind = "target" if state in env.target_states else "forbidden" if state in env.forbidden_states else "normal"
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor=GRID_COLORS[kind],
                               edgecolor="black", linewidth=1, alpha=.7))
        actions = np.flatnonzero(np.isclose(policy[state], np.max(policy[state])))
        arrow = ARROWS[env.actions[int(actions[0])]]
        ax.text(col + .5, row + .48, str(state), ha="center", va="center",
                fontsize=10, color="black", fontweight="bold", zorder=6,
                bbox={"facecolor": "white", "alpha": .72, "edgecolor": "none", "pad": .5})
        ax.text(col + .5, row + .20, f"{values[state]:.2f}", ha="center", va="center",
                fontsize=8, color="blue", zorder=6)
        ax.text(col + .5, row + .76, arrow, ha="center", va="center",
                fontsize=15, color="black", fontweight="bold", zorder=6)
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("GridWorld policy, values and agent trajectory")
    ax.legend(handles=[Patch(facecolor=GRID_COLORS[k], edgecolor="black", label=label)
                       for k, label in (("normal", "Normal"), ("forbidden", "Forbidden"), ("target", "Target"))],
              loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    return fig, ax


def _path_xy(path, size):
    return [state % size + .5 for state in path], [state // size + .5 for state in path]


def grid_figure(result, step=None):
    fig, ax = _grid_canvas(result)
    path = result["path"][: (step + 1) if step is not None else len(result["path"])]
    if path:
        cols, rows = _path_xy(path, result["env"].size)
        ax.plot(cols, rows, color="#d7191c", linewidth=2, marker="o", markersize=3, alpha=.8, zorder=4)
        ax.scatter([cols[-1] + .28], [rows[-1] - .28], s=150, marker="*",
                   c="#d7191c", edgecolors="black", linewidths=.5, zorder=8)
    return fig


def estimation_figures(result):
    samples, trajectory, target = result["samples"], result["trajectory"], result["target"]
    fig, (ax, err_ax) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax.scatter(samples[:, 0], samples[:, 1], s=12, alpha=.35, label="Samples")
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="#d7191c", linewidth=1.6, label="Parameter path")
    ax.scatter(*trajectory[0], marker="s", color="#1f77b4", label="Start")
    ax.scatter(*target, marker="*", s=180, color="#2ca02c", label=result.get("target_label", "Sample mean"))
    ax.set_title("Optimizer path in parameter space")
    ax.set_xlabel("w₁")
    ax.set_ylabel("w₂")
    ax.legend(fontsize=8)
    err_ax.plot(result["errors"], color="#7b3294")
    err_ax.set_title("Distance to the sample mean")
    err_ax.set_xlabel("Update")
    err_ax.set_ylabel("Euclidean error")
    err_ax.grid(alpha=.25)
    fig.tight_layout()
    return fig


def save_exploration_gif(result, output_path, fps=3, title="Agent exploration"):
    """Animate a recorded episode without changing the chapter algorithm."""
    path = result["path"]
    if not path:
        raise ValueError("A GridWorld animation needs at least one recorded state")
    fig, ax = _grid_canvas(result)
    cols, rows = _path_xy(path, result["env"].size)
    trail, = ax.plot([], [], color="#d7191c", linewidth=2, marker="o", markersize=3,
                     alpha=.8, zorder=4)
    agent = ax.scatter([], [], s=150, marker="*", c="#d7191c", edgecolors="black",
                       linewidths=.5, zorder=8)
    steps = np.unique(np.linspace(0, len(path) - 1, min(24, len(path)), dtype=int)).tolist()
    frames = steps + [steps[-1]] * 3

    def update(frame):
        step = frames[frame]
        trail.set_data(cols[:step + 1], rows[:step + 1])
        agent.set_offsets([[cols[step] + .28, rows[step] - .28]])
        ax.set_title(f"{title} · step {step}/{len(path) - 1} · state S{path[step]}")
        return trail, agent

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(fig)
    return output


def save_estimation_gif(result, output_path, fps=3, title="SGD"):
    """Animate Chapter 5's parameter path and estimation error."""
    trajectory, errors = result["trajectory"], result["errors"]
    fig, (ax, err_ax) = plt.subplots(1, 2, figsize=(10, 4.5))
    samples = result["samples"]
    ax.scatter(samples[:, 0], samples[:, 1], s=10, alpha=.2, color="#1f77b4")
    ax.scatter(*result["target"], marker="*", s=180, color="#2ca02c", label=result.get("target_label", "Sample mean"))
    ax.set_xlim(min(samples[:, 0].min(), trajectory[:, 0].min()) - 2, max(samples[:, 0].max(), trajectory[:, 0].max()) + 2)
    ax.set_ylim(min(samples[:, 1].min(), trajectory[:, 1].min()) - 2, max(samples[:, 1].max(), trajectory[:, 1].max()) + 2)
    ax.set_xlabel("w₁")
    ax.set_ylabel("w₂")
    ax.legend(fontsize=8)
    err_ax.set_xlim(0, max(1, len(errors) - 1))
    err_ax.set_ylim(0, max(1, float(np.max(errors)) * 1.1))
    err_ax.set_xlabel("Update")
    err_ax.set_ylabel(f"Distance to {result.get('target_label', 'Sample mean').lower()}")
    err_ax.grid(alpha=.25)
    path_line, = ax.plot([], [], color="#d7191c", linewidth=2)
    current = ax.scatter([], [], s=80, color="#d7191c", zorder=5)
    error_line, = err_ax.plot([], [], color="#7b3294", linewidth=2)
    steps = np.unique(np.linspace(0, len(trajectory) - 1, min(24, len(trajectory)), dtype=int)).tolist()
    frames = steps + [steps[-1]] * 3

    def update(frame):
        step = frames[frame]
        path_line.set_data(trajectory[:step + 1, 0], trajectory[:step + 1, 1])
        current.set_offsets(trajectory[step:step + 1])
        error_line.set_data(np.arange(step + 1), errors[:step + 1])
        ax.set_title(f"{title} · update {step}/{len(trajectory) - 1}")
        err_ax.set_title(f"Error {errors[step]:.3f}")
        return path_line, current, error_line

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return output
