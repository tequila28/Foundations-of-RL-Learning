# 第 9 章：以 GSM8K 理解 LLM 中的 PPO 与 GRPO

[English](README_en.md) | [简体中文](README.md)

本章把前八章的策略梯度、价值函数与优势估计连接到语言模型。以 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 和 [GSM8K](https://github.com/openai/grade-school-math) 为例，分别讲解 PPO、GRPO 从 prompt 采样、奖励计算、优势估计到策略更新的训练原理与代码流程。

## 从 prompt 到策略更新：训练流程动图

下面两张动图分别演示 PPO 和 GRPO 的训练流程：回答逐步生成，随后计算奖励与优势，最后通过损失函数反向传播并更新策略。PPO 对一个 prompt 采样一个回答；GRPO 对同一 prompt 采样一组回答。

**PPO：** 一个 prompt 采样一个回答；最后的答案得到 0/1 奖励，critic（代码中的价值头）给生成过程中的各个状态估计价值。GAE 将终点奖励和这些价值估计结合，从后往前计算每个 token 的优势；再由 token logits 得到对数概率，构造裁剪损失并反向传播，更新策略与价值头。

![PPO 从 prompt、单个回答、奖励与价值估计到策略更新的动态示意](assets/ppo_training_flow.gif)

**GRPO：** 同一 prompt 生成一组回答，分别打分并计算组内相对优势；用裁剪目标和参考策略 KL 更新策略，不训练价值头。

![GRPO 从 prompt、多个回答、组内奖励比较到策略更新的动态示意](assets/grpo_training_flow.gif)

## 学习目标与实验任务

GSM8K 每条记录包含 `question` 和 `answer`。标准答案以 `#### 数值` 结尾；提示词要求模型按相同格式给出最终答案。本章使用最简单的规则奖励：

```text
R(x, y) = 1，若生成答案的最终数值与标准答案一致
        = 0，其他情况
```

这里的 `x` 是题目，`y` 是生成的推理与答案。只有训练集参与策略更新；测试集只用于最终答案准确率评估。奖励只检查最终数值，不判断推理过程是否正确。

## PPO：用价值头估计每个 token 的优势

PPO 在当前策略下为一道题生成一个回答。生成时保存旧策略的 token 对数概率；冻结的初始模型作为参考策略。任务奖励放在回答的最后一个有效 token，其他 token 可加入相对参考策略的 KL 代价：

```text
r_t = -β [log π_old(a_t | s_t) - log π_ref(a_t | s_t)]
      + 1[t = T] · R(x, y)
```

语言模型最后一层隐藏状态接一个线性价值头，即 critic，用来估计生成过程中每个状态的价值 `V(s_t)`。终点的任务奖励和每个 token 的 KL 代价组成 `r_t`；GAE 从后往前计算各 token 的优势与回报目标：

```text
δ_t = r_t + γ V(s_{t+1}) - V(s_t)     （终点后状态价值为 0）
A_t = δ_t + γλ A_{t+1}                （从最后一个 token 向前递推）
G_t = A_t + V(s_t)
```

策略更新比较当前策略和生成回答时保存的旧策略，并裁剪概率比；价值头同时拟合回报目标：

```text
ρ_t   = exp[log π_θ(a_t | s_t) - log π_old(a_t | s_t)]
L_PPO = -mean_t min[ρ_t A_t, clip(ρ_t, 1−ε, 1+ε) A_t]
        + (c_v / 2) · mean_t [V_θ(s_t) − G_t]²
```

`src/ppo.py` 展示“生成 → 奖励/KL → GAE → 策略与价值更新 → 保存”的完整路径。PPO 需要额外训练价值头。

## GRPO：同题多答案的相对奖励

GRPO 对同一道题采样 `G` 个回答，各自计算规则奖励，然后在组内求优势：

```text
平均奖励 R̄ = mean(R_1, ..., R_G)
组内优势 A_i = (R_i − R̄) / [std(R_1, ..., R_G) + ε]
```

这个优势赋给该回答的每个有效 token。策略损失同样使用旧策略概率比和裁剪，并加入相对冻结参考策略的 KL 项；先对每个回答的 token 求平均，再对同题回答组求平均：

```text
ρ_i,t    = exp[log π_θ(a_i,t | s_i,t) − log π_old(a_i,t | s_i,t)]
L_GRPO   = −mean_i mean_t min[ρ_i,t A_i, clip(ρ_i,t, 1−ε, 1+ε) A_i]
           + β · mean_i mean_t KL_i,t
Δ_i,t    = log π_ref(a_i,t | s_i,t) − log π_θ(a_i,t | s_i,t)
KL_i,t   = exp(Δ_i,t) − Δ_i,t − 1
```

GRPO 不训练价值头。若一组回答的奖励完全相同，其组内优势为零，因此奖励设计和回答的多样性会影响更新信号。

`src/grpo.py` 实现“同题多次生成 → 组内奖励与优势 → 裁剪更新 → 保存”的流程。

## 目录结构

```text
Chapter9_LLM_RL_GSM8K/
├── README.md / README_en.md     # 中英文理论与使用说明
├── requirements.txt             # 仅声明 torch、transformers
├── assets/                      # PPO / GRPO 训练流程 GIF
├── data/                        # 下载的 train.jsonl、test.jsonl（不提交）
├── outputs/                     # 策略权重与评估输出（不提交）
├── scripts/
│   ├── download_gsm8k.py        # 标准库下载原始 JSONL
│   ├── train_ppo.sh
│   ├── train_grpo.sh
│   └── evaluate.sh
└── src/
    ├── data.py                  # JSONL 读取
    ├── reward.py                # #### 最终答案与规则奖励
    ├── modeling.py              # 提示词、采样、token 概率与保存
    ├── ppo.py                   # 价值头、GAE、PPO 更新
    ├── grpo.py                  # 组内优势与 GRPO 更新
    └── evaluate.py              # 测试集精确答案准确率
```

## 使用方式

从**仓库根目录**运行。新章节的直接依赖只有 PyTorch 和 Transformers，下载脚本使用 Python 标准库；不需要 `datasets`、`trl`、`accelerate` 或 `peft`。

```bash
pip install -r Chapter9_LLM_RL_GSM8K/requirements.txt
python Chapter9_LLM_RL_GSM8K/scripts/download_gsm8k.py

python -m Chapter9_LLM_RL_GSM8K.src.ppo --steps 20
python -m Chapter9_LLM_RL_GSM8K.src.grpo --steps 20 --group-size 4

python -m Chapter9_LLM_RL_GSM8K.src.evaluate \
  --model Chapter9_LLM_RL_GSM8K/outputs/grpo \
  --predictions Chapter9_LLM_RL_GSM8K/outputs/grpo_test_predictions.jsonl
```

默认模型为 `Qwen/Qwen2.5-0.5B-Instruct`，训练入口默认只读取前 64 道训练题、更新 20 次。可以通过 `--model` 指向本地模型目录，通过 `--max-examples`、`--max-new-tokens`、`--learning-rate` 等参数调整演示。PPO 保存策略与 `value_head.pt`，GRPO 只保存策略；评估只需加载策略目录。Bash 脚本是等价入口，例如 `bash Chapter9_LLM_RL_GSM8K/scripts/train_grpo.sh --steps 20`。

## 运行与实现提示

- 直接更新 0.5B 模型并保留参考模型会占用较多内存；代码未加入 LoRA、分布式训练或显存优化。
- 本章使用最终答案奖励。若 GRPO 的同题回答全部得到相同分数，组内优势为零，此次更新没有来自相对奖励的策略信号。
- 生成时可能提前遇到 EOS；代码只把首个 EOS 计入有效 token，忽略随后用于对齐长度的填充 token。

## 参考

- [GSM8K 官方仓库与数据格式](https://github.com/openai/grade-school-math)
- [Qwen2.5-0.5B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [PPO 原论文](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath / GRPO 原论文](https://arxiv.org/abs/2402.03300)
