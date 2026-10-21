"""Streamlit entry point for the Foundations of RL interactive lab."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import streamlit as st

from apps.gridworld_lab.runner import CHAPTERS, run
from apps.gridworld_lab.visualize import estimation_figures, grid_figure, save_estimation_gif, save_exploration_gif


st.set_page_config(page_title="RL Learning Lab", page_icon="🧭", layout="wide")
st.title("🧭 Foundations of RL · Interactive Learning Lab")
st.caption("选择章节和算法，在一个面板里调整参数并观察策略、状态价值、训练回报与智能体轨迹。")

with st.sidebar:
    st.header("实验设置")
    chapter = st.selectbox("学习章节", list(CHAPTERS))
    algorithm = st.selectbox("算法", CHAPTERS[chapter])
    seed = st.number_input("随机种子", min_value=0, max_value=999999, value=42)
    st.subheader("训练参数")
    episodes = st.slider("训练回合数", min_value=10, max_value=1000, value=200, step=10)
    max_steps = st.slider("每回合最大步数", min_value=5, max_value=100, value=40, step=5)
    alpha = st.slider("学习率 α", min_value=0.001, max_value=1.0, value=0.1, step=0.001)
    epsilon = st.slider("探索率 ε", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    gamma = st.slider("折扣因子 γ", min_value=0.0, max_value=0.99, value=0.9, step=0.01)
    if chapter.startswith("第 3 章"):
        iterations = st.slider("DP 最大迭代次数", 10, 2000, 500, 10)
        theta = st.select_slider("收敛阈值 θ", options=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6], value=1e-5, format_func=lambda x: f"{x:g}")
        inner_iterations = st.slider("截断评估步数", 1, 100, 10)
    else:
        iterations, theta, inner_iterations = 500, 1e-5, 10
    if chapter.startswith("第 5 章"):
        samples = st.slider("样本数", 50, 2000, 400, 50)
        batch_size = st.slider("小批量大小", 1, 256, 32)
    else:
        samples, batch_size = 400, 32
    if chapter.startswith("第 8 章"):
        hidden_size = st.select_slider("隐藏层宽度", options=[16, 32, 64, 128], value=64)
    else:
        hidden_size = 64

    if chapter.startswith("第 5 章"):
        size, target, start, forbidden = 5, 17, 0, []
        r_step, r_forbid, r_target, r_wall = -0.04, -1.0, 1.0, -0.04
    else:
        st.subheader("网格环境")
        size = st.slider("网格边长", min_value=3, max_value=8, value=5)
        valid_states = list(range(size * size))
        target_default = min(17, size * size - 1)
        target = st.selectbox("目标状态", valid_states, index=target_default)
        start = st.selectbox("智能体起点", valid_states, index=0)
        forbidden_default = [s for s in [6, 7, 12, 16, 18, 21] if s < size * size and s != target]
        forbidden = st.multiselect("惩罚状态", valid_states, default=forbidden_default)
        r_step = st.number_input("普通步奖励", value=-0.04, step=0.01)
        r_forbid = st.number_input("惩罚状态奖励", value=-1.0, step=0.1)
        r_target = st.number_input("到达目标奖励", value=1.0, step=0.1)
        r_wall = st.number_input("撞墙奖励", value=-0.04, step=0.01)

    run_button = st.button("运行实验", type="primary", use_container_width=True)

if run_button:
    config = {
        "seed": int(seed), "size": size, "gamma": gamma, "target": target, "start": start,
        "forbidden": forbidden, "r_step": r_step, "r_forbid": r_forbid,
        "r_target": r_target, "r_wall": r_wall, "episodes": episodes,
        "max_steps": max_steps, "alpha": alpha, "epsilon": epsilon,
        "iterations": iterations, "theta": theta, "inner_iterations": inner_iterations,
        "samples": samples, "batch_size": batch_size, "hidden_size": hidden_size,
    }
    try:
        with st.spinner("正在运行算法并生成结果…"):
            st.session_state["rl_lab_result"] = run(chapter, algorithm, config)
            st.session_state["rl_lab_label"] = (chapter, algorithm)
            st.session_state.pop("rl_lab_gif", None)
    except Exception as exc:
        st.error(f"实验运行失败：{exc}")

result = st.session_state.get("rl_lab_result")
if result is None:
    st.info("从左侧选择章节、算法和参数，然后点击“运行实验”。")
else:
    old_chapter, old_algorithm = st.session_state.get("rl_lab_label", (chapter, algorithm))
    st.subheader(f"{old_chapter} / {old_algorithm}")
    if result["kind"] == "estimation":
        left, right = st.columns([2, 1])
        with left:
            st.pyplot(estimation_figures(result), use_container_width=True)
        with right:
            st.metric("最终估计值", f"({result['trajectory'][-1, 0]:.3f}, {result['trajectory'][-1, 1]:.3f})")
            st.metric("样本均值", f"({result['target'][0]:.3f}, {result['target'][1]:.3f})")
            st.metric("最终误差", f"{result['errors'][-1]:.4f}")
        st.caption("第 5 章研究随机优化器的采样与收敛过程，因此此处展示原实验中的二维均值估计任务。")
    else:
        path = result["path"]
        current_step = st.slider("回放智能体轨迹", min_value=0, max_value=max(0, len(path) - 1), value=max(0, len(path) - 1))
        left, right = st.columns([3, 2])
        with left:
            st.pyplot(grid_figure(result, current_step), use_container_width=True)
        with right:
            st.metric("当前状态", f"S{path[current_step]}  (第 {current_step} 步)")
            st.metric("估值范围", f"{result['values'].min():.3f} 至 {result['values'].max():.3f}")
            if result["rewards"]:
                st.line_chart(result["rewards"], height=220, x_label="Episode", y_label="Return")
                window = min(20, len(result["rewards"]))
                st.caption(f"最近 {window} 回合平均回报：{sum(result['rewards'][-window:]) / window:.3f}")
            else:
                st.write("该章节展示规划得到的状态价值、策略方向和一次轨迹回放。")
        st.caption("白色、橙色、浅蓝色分别表示普通、惩罚、目标状态；蓝色数值表示状态价值或动作价值摘要，箭头表示策略动作，红线表示智能体轨迹。")

    if st.button("生成当前实验 GIF"):
        with st.spinner("正在逐帧生成动图…"):
            with TemporaryDirectory() as temporary_dir:
                gif_path = Path(temporary_dir) / "rl_lab_experiment.gif"
                if result["kind"] == "estimation":
                    save_estimation_gif(result, gif_path, title=old_algorithm)
                else:
                    save_exploration_gif(result, gif_path)
                st.session_state["rl_lab_gif"] = gif_path.read_bytes()
    if st.session_state.get("rl_lab_gif"):
        st.image(st.session_state["rl_lab_gif"], caption="当前参数下的运行过程", use_column_width=True)
        st.download_button("下载当前实验 GIF", data=st.session_state["rl_lab_gif"],
                           file_name="rl_lab_experiment.gif", mime="image/gif")

plt.close("all")
