# Interactive RL Learning Lab

This cross-chapter Streamlit app provides one parameter panel for the repository's eight chapters. It displays GridWorld policies, state values, episode returns, and a replayable agent path. The "生成当前实验 GIF" button generates a downloadable animation for the current parameters. Chapter 5 keeps its stochastic-approximation mean-estimation task and plots optimizer paths and errors.

## Run

From the repository root, install the requirements and start the app:

```bash
pip install -r requirements.txt
streamlit run apps/gridworld_lab/app.py
```

## Layout

- `app.py` defines chapter selection, parameter controls, and result panels.
- `runner.py` maps chapter selections to independent modules in `chapters/`.
- `common.py` contains shared environment, policy, and rollout helpers.
- `chapters/chapter1.py` through `chapter8.py` implement one visualization experiment per chapter.
- `visualize.py` renders the shared GridWorld and optimization figures.
- `generate_previews.py` regenerates optional static PNG previews without touching the original experiment results.
- `chapter_defaults.py` reads the default parameter assignments from each chapter's Bash script.
- `generate_chapter_gifs.py` runs the representative experiments with those defaults and regenerates the eight GIF animations embedded in the chapter READMEs.
- `generate_agent_gif.py` produces the ε-greedy Q-learning exploration animation used in the root and Chapter 1 READMEs.

Add an algorithm name to `CHAPTERS`, implement its experiment in the matching module under `chapters/`, and return the display data expected by `app.py`. Grid experiments return a policy, a value vector, a trajectory, and optional episode returns. Optimizer experiments return samples, a parameter path, a target, and error history.

Regenerate all eight chapter GIF animations from the repository root with:

```bash
python -m apps.gridworld_lab.generate_chapter_gifs
```

The generator reads the selected experiment's Bash defaults. For reproducible replay, it uses seed 42 and start state S0 when the script does not specify them. Chapters 1–3 do not define an episode length, so their animations replay 25 steps. Chapter 7 uses the 500-step episode length defined in its first experiment's Python entry point.

Regenerate optional static PNG previews with:

```bash
python -m apps.gridworld_lab.generate_previews
```

Regenerate the Q-learning exploration animation with:

```bash
python -m apps.gridworld_lab.generate_agent_gif
```
