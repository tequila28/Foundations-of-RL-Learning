# 第 9 章：以 GSM8K 理解 LLM 中的 PPO 与 GRPO

[English](README_en.md) | [简体中文](README.md)

本章把前八章的策略梯度、价值函数与优势估计连接到语言模型。默认策略是 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，问题来自 [GSM8K 原始数据集](https://github.com/openai/grade-school-math)。代码展示**教学用训练流程**，未在本项目中完成模型下载、完整训练或效果验证。

## 从 prompt 到策略更新：动态示意

下面两张 GIF 由本项目绘制，采用浅米色背景和低饱和度素描线条。回答会逐步生成，奖励、价值与优势随后出现，梯度再返回策略；PPO 展示单个回答，GRPO 展示同题多个回答。示例题目、回答和数值仅用于解释流程，**不是模型真实生成结果或训练曲线**。

**PPO：** 一个 prompt 采样一个回答；最后的答案得到 0/1 奖励，critic（代码中的价值头）给生成过程中的各个状态估计价值。GAE 将终点奖励和这些价值估计结合，从后往前计算每个 token 的优势；再由 token logits 得到对数概率，构造裁剪损失并反向传播，更新策略与价值头。

![PPO 从 prompt、单个回答、奖励与价值估计到策略更新的动态示意](assets/ppo_training_flow.gif)

**GRPO：** 同一 prompt 生成一组回答，分别打分并计算组内相对优势；用裁剪目标和参考策略 KL 更新策略，不训练价值头。

![GRPO 从 prompt、多个回答、组内奖励比较到策略更新的动态示意](assets/grpo_training_flow.gif)

## 学习目标与实验任务

GSM8K 每条记录包含 `question` 和 `answer`。标准答案以 `#### 数值` 结尾；提示词要求模型按相同格式给出最终答案。本章使用最简单的规则奖励：

$$
R(x,y)=\begin{cases}
1,&\text{生成答案与标准答案的最终数值相同},\\
0,&\text{其他情况}.
\end{cases}
$$

这里的 `x` 是题目，`y` 是生成的推理与答案。只有训练集参与策略更新；测试集只用于最终答案准确率评估。奖励只检查最终数值，不判断推理过程是否正确。

## PPO：用价值头估计每个 token 的优势

PPO 在当前策略下为一道题生成一个回答。生成时保存旧策略的 token 对数概率；冻结的初始模型作为参考策略。任务奖励放在回答的最后一个有效 token，其他 token 可加入相对参考策略的 KL 代价：

$$
r_t=-\beta\bigl(\log\pi_{\mathrm{old}}(a_t\mid s_t)
-\log\pi_{\mathrm{ref}}(a_t\mid s_t)\bigr)
+\mathbf{1}_{t=T}R(x,y).
$$

语言模型最后一层隐藏状态接一个线性价值头 `V(s_t)`。代码用 GAE 从后往前计算优势 `A_t` 与回报目标，并对同一批生成结果做多轮更新：

$$
\rho_t=\exp\bigl(\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\mathrm{old}}(a_t\mid s_t)\bigr),
\qquad
L_{\mathrm{PPO}}=
-\mathbb{E}_t\!\left[
\min\left(\rho_t A_t,\,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)
\right]
+c_v\,\frac{1}{2}\mathbb{E}_t[(V_\theta(s_t)-G_t)^2].
$$

`src/ppo.py` 展示“生成 → 奖励/KL → GAE → 策略与价值更新 → 保存”的完整路径。PPO 需要额外训练价值头。

## GRPO：同题多答案的相对奖励

GRPO 对同一道题采样 `G` 个回答，各自计算规则奖励，然后在组内求优势：

$$
A_i=\frac{R_i-\bar R}{\operatorname{std}(R_1,\ldots,R_G)+\varepsilon}.
$$

这个优势赋给该回答的有效 token。策略损失同样使用旧策略概率比和裁剪，并加入相对冻结参考策略的 KL 项。每个回答先按其 token 数平均，再对整组回答平均。**GRPO 不训练价值头**；若一组回答的奖励完全相同，该组的相对优势为零，说明稀疏的 0/1 奖励可能不足以提供学习信号。

`src/grpo.py` 展示“同题多次生成 → 组内奖励与优势 → 裁剪更新 → 保存”。这里是便于理解的最小实现，不等同于大型训练框架中的全部工程细节。

## 目录结构

```text
Chapter9_LLM_RL_GSM8K/
├── README.md / README_en.md     # 中英文理论与使用说明
├── requirements.txt             # 仅声明 torch、transformers
├── assets/                      # PPO / GRPO 训练流程示意 GIF
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

## 范围与注意

- 这是完整训练步骤的教学示例，但**未做端到端训练验证**，没有声称达到任何 GSM8K 分数。
- 直接更新 0.5B 模型和保留一份参考模型会占用较多内存；示例没有实现 LoRA、分布式训练或显存优化。
- 本章只用最终答案奖励。起点模型若很少答对，尤其是 GRPO 的同题组可能全为 0，学习信号会很弱；正式实验通常需要更合适的初始模型、奖励设计和训练配置。
- 生成时可能提前遇到 EOS；代码只把首个 EOS 计入有效 token，忽略随后用于对齐长度的填充 token。

## 参考

- [GSM8K 官方仓库与数据格式](https://github.com/openai/grade-school-math)
- [Qwen2.5-0.5B-Instruct 模型卡](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [PPO 原论文](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath / GRPO 原论文](https://arxiv.org/abs/2402.03300)
