"""Matplotlib visualizations shared by the interactive learning lab."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path


ARROWS = {"up": "↑", "right": "→", "down": "↓", "left": "←", "stay": "·"}


def grid_figure(result, step=None):
    env, policy, values = result["env"], result["policy"], result["values"]
    path = result["path"]
    value_label = result.get("value_label", "V(s)")
    cell_value_label = result.get("cell_value_label", value_label)
    size = env.size
    fig, ax = plt.subplots(figsize=(7, 6))
    vals = values.reshape(size, size)
    im = ax.imshow(vals, cmap="YlGnBu", origin="upper")
    fig.colorbar(im, ax=ax, shrink=0.78, label=value_label)
    for s in range(env.n_states):
        row, col = divmod(s, size)
        cell_color = "#f3a6a6" if s in env.forbidden_states else "#91d8a7" if s in env.target_states else None
        if cell_color:
            ax.add_patch(Rectangle((col - .5, row - .5), 1, 1, color=cell_color, alpha=.76))
        action_probs = policy[s]
        best = np.flatnonzero(np.isclose(action_probs, np.max(action_probs)))
        arrows = "".join(ARROWS[env.actions[a]] for a in best[:4])
        label = f"S{s}\n{cell_value_label}={values[s]:.2f}\n{arrows}"
        if s in env.target_states:
            label += "\nGOAL"
        elif s in env.forbidden_states:
            label += "\nPENALTY"
        span = max(float(np.ptp(values)), 1e-8)
        text_color = "white" if values[s] > values.min() + 0.68 * span and not cell_color else "#15202b"
        ax.text(col, row, label, ha="center", va="center", fontsize=8, color=text_color, fontweight="medium", zorder=6,
                bbox={"facecolor": "white", "alpha": .68, "edgecolor": "none", "pad": .7})
    visible = path[: (step + 1) if step is not None else len(path)]
    if visible:
        rows = [s // size for s in visible]
        cols = [s % size for s in visible]
        ax.plot(cols, rows, color="#d7191c", linewidth=2, marker="o", markersize=3, alpha=.82, zorder=2)
        ax.scatter([cols[-1]], [rows[-1]], s=140, marker="*", c="#ff3b30", edgecolors="white", zorder=5)
    ax.set_xticks(range(size), labels=range(size))
    ax.set_yticks(range(size), labels=range(size))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("GridWorld policy, values and agent trajectory")
    ax.set_xticks(np.arange(-.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, size, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    fig.tight_layout()
    return fig


def estimation_figures(result):
    samples, trajectory, target = result["samples"], result["trajectory"], result["target"]
    fig, (ax, err_ax) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax.scatter(samples[:, 0], samples[:, 1], s=12, alpha=.35, label="Samples")
    ax.plot(trajectory[:, 0], trajectory[:, 1], color="#d7191c", linewidth=1.6, label="Parameter path")
    ax.scatter(*trajectory[0], marker="s", color="#1f77b4", label="Start")
    ax.scatter(*target, marker="*", s=180, color="#2ca02c", label="Sample mean")
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


def save_exploration_gif(result, output_path, fps=3):
    """Save a replayable animation of one recorded GridWorld episode."""
    env, policy, values = result["env"], result["policy"], result["values"]
    path = result["path"]
    value_label = result.get("value_label", "V(s)")
    cell_value_label = result.get("cell_value_label", value_label)
    size = env.size
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(values.reshape(size, size), cmap="YlGnBu", origin="upper")
    fig.colorbar(image, ax=ax, shrink=0.78, label=value_label)
    for state in range(env.n_states):
        row, col = divmod(state, size)
        special_color = "#f3a6a6" if state in env.forbidden_states else "#91d8a7" if state in env.target_states else None
        if special_color:
            ax.add_patch(Rectangle((col - .5, row - .5), 1, 1, color=special_color, alpha=.9))
        best_actions = np.flatnonzero(np.isclose(policy[state], np.max(policy[state])))
        arrows = "".join(ARROWS[env.actions[action]] for action in best_actions[:4])
        label = f"S{state}\n{cell_value_label}={values[state]:.2f}\n{arrows}"
        if state in env.target_states:
            label += "\nGOAL"
        elif state in env.forbidden_states:
            label += "\nPENALTY"
        value_span = max(float(np.ptp(values)), 1e-8)
        color = "white" if values[state] > values.min() + .68 * value_span and not special_color else "#15202b"
        ax.text(col, row, label, ha="center", va="center", fontsize=8, color=color, fontweight="medium", zorder=6,
                bbox={"facecolor": "white", "alpha": .68, "edgecolor": "none", "pad": .7})

    ax.set_xticks(range(size), labels=range(size))
    ax.set_yticks(range(size), labels=range(size))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(-.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, size, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    rows = [state // size for state in path]
    cols = [state % size for state in path]
    trail, = ax.plot([], [], color="#d7191c", linewidth=2.5, marker="o", markersize=3, alpha=.85)
    agent = ax.scatter([], [], s=180, marker="*", c="#ffcc00", edgecolors="#15202b", linewidths=.8, zorder=5)
    title = ax.set_title("")
    frames = list(range(len(path))) + [len(path) - 1] * 3

    def update(frame):
        step = frames[frame]
        trail.set_data(cols[:step + 1], rows[:step + 1])
        agent.set_offsets([[cols[step] + .30, rows[step] + .30]])
        title.set_text(f"Q-learning ε-greedy exploration · step {step}/{len(path) - 1} · state S{path[step]}")
        return trail, agent, title

    animation = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False, repeat=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    return output
