# Chapter 9: Understanding PPO and GRPO for LLMs with GSM8K

[English](README_en.md) | [简体中文](README.md)

This chapter connects the policy gradients, value functions, and advantage estimates of the earlier chapters to language models. The default policy is [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), and the questions come from the [original GSM8K dataset](https://github.com/openai/grade-school-math). This is an **educational training flow**: model download, full training, and resulting accuracy have not been verified in this project.

## From a prompt to a policy update: animated diagrams

These two GIFs were drawn for this project in a light beige, low-saturation sketch style. The answer pieces move into place, followed by reward, value and advantage estimates, and a gradient moving back toward the policy. PPO shows one answer; GRPO shows several answers to the same prompt. The sample question, completions, and numbers are **illustrative, not actual model generations or training results**.

**PPO:** Sample one answer for one prompt. The final answer receives a 0/1 reward, while the critic (a value head in this code) estimates values for intermediate generated states. GAE combines the terminal reward with these estimates to calculate each token's advantage backward through the response. Token logits yield log probabilities for the clipped loss; backpropagation updates both policy and value head.

![Animated PPO flow from a prompt through one answer, reward, value estimation, and policy update](assets/ppo_training_flow.gif)

**GRPO:** Generate a group of answers to the same prompt, score each answer, compute group-relative advantages, and update the policy with clipping and reference-policy KL—without training a value head.

![Animated GRPO flow from a prompt through grouped answers, relative rewards, and policy update](assets/grpo_training_flow.gif)

## Task and reward

Each GSM8K record has a `question` and an `answer`. The reference answer ends with `#### number`, and the prompt asks the model to use the same final-answer format. The rule-based reward is deliberately simple:

$$
R(x,y)=\begin{cases}
1,&\text{the final numbers in the generated and reference answers match},\\
0,&\text{otherwise}.
\end{cases}
$$

Here `x` is a question and `y` is a generated solution. Only training records update the policy; the test split is reserved for final-answer accuracy. The reward checks the final number, not the validity of the intermediate reasoning.

## PPO: a value head and token-level advantages

PPO samples one answer per question under the current policy. It stores the old policy's token log probabilities, and keeps a frozen copy of the initial model as a reference. The answer reward is added to the final valid token, while each token may receive a KL cost relative to the reference policy:

$$
r_t=-\beta\bigl(\log\pi_{\mathrm{old}}(a_t\mid s_t)
-\log\pi_{\mathrm{ref}}(a_t\mid s_t)\bigr)
+\mathbf{1}_{t=T}R(x,y).
$$

A linear value head on the language model's last hidden state estimates `V(s_t)`. Generalized Advantage Estimation (GAE) works backward over generated tokens to construct advantages `A_t` and return targets. The sampled answer then supports several clipped updates:

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

`src/ppo.py` covers generation, reward and KL construction, GAE, policy and value updates, and saving. PPO trains the extra value head.

## GRPO: relative rewards within a question

GRPO samples `G` answers to the **same** question and compares their rewards within that group:

$$
A_i=\frac{R_i-\bar R}{\operatorname{std}(R_1,\ldots,R_G)+\varepsilon}.
$$

Each valid token in an answer receives its answer's advantage. The policy objective uses the old-policy probability ratio, clipping, and a KL term against the frozen reference. The implementation averages over tokens within each answer and then over the group. **GRPO has no value head.** If all answers in a group have the same reward, their relative advantages are zero; sparse binary rewards can therefore give a weak learning signal.

`src/grpo.py` covers grouped generation, reward comparison, clipped updates, and saving. It is a minimal teaching implementation rather than a full large-scale training framework.

## Files

```text
Chapter9_LLM_RL_GSM8K/
├── README.md / README_en.md     # Theory and usage in Chinese and English
├── requirements.txt             # Only torch and transformers
├── assets/                      # PPO / GRPO training-flow GIFs
├── data/                        # Downloaded train.jsonl and test.jsonl (ignored)
├── outputs/                     # Saved policies and predictions (ignored)
├── scripts/
│   ├── download_gsm8k.py        # Download original JSONL using the standard library
│   ├── train_ppo.sh
│   ├── train_grpo.sh
│   └── evaluate.sh
└── src/
    ├── data.py                  # JSONL loading
    ├── reward.py                # #### final-answer extraction and reward
    ├── modeling.py              # Prompts, sampling, token scores, saving
    ├── ppo.py                   # Value head, GAE, PPO update
    ├── grpo.py                  # Group advantages and GRPO update
    └── evaluate.py              # Exact final-answer accuracy on the test split
```

## Usage

Run these commands from the **repository root**. This chapter directly requires only PyTorch and Transformers; the download script uses the Python standard library. It does not require `datasets`, `trl`, `accelerate`, or `peft`.

```bash
pip install -r Chapter9_LLM_RL_GSM8K/requirements.txt

python Chapter9_LLM_RL_GSM8K/scripts/download_gsm8k.py

python -m Chapter9_LLM_RL_GSM8K.src.ppo --steps 20
python -m Chapter9_LLM_RL_GSM8K.src.grpo --steps 20 --group-size 4

python -m Chapter9_LLM_RL_GSM8K.src.evaluate \
  --model Chapter9_LLM_RL_GSM8K/outputs/grpo \
  --predictions Chapter9_LLM_RL_GSM8K/outputs/grpo_test_predictions.jsonl
```

The default model is `Qwen/Qwen2.5-0.5B-Instruct`. Training defaults to the first 64 training questions and 20 updates. Options such as `--model`, `--max-examples`, `--max-new-tokens`, and `--learning-rate` control the demo. PPO saves the policy plus `value_head.pt`; GRPO saves the policy only. Evaluation loads only the policy. The Bash wrappers provide equivalent entry points, for example `bash Chapter9_LLM_RL_GSM8K/scripts/train_grpo.sh --steps 20`.

## Scope and limitations

- The training steps are written out, but **end-to-end training has not been tested**, and no GSM8K score is claimed.
- Updating a 0.5B model while keeping a reference copy needs substantial memory. This example does not implement LoRA, distributed training, or memory optimization.
- With a final-answer-only reward, a weak initial model may rarely answer correctly. In particular, GRPO groups may all score zero. Serious experiments need an appropriate starting model, reward design, and training configuration.
- If generation reaches EOS early, only the first EOS is treated as a valid token; subsequent padding is ignored.

## References

- [Official GSM8K repository and data format](https://github.com/openai/grade-school-math)
- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Original PPO paper](https://arxiv.org/abs/1707.06347)
- [DeepSeekMath / GRPO paper](https://arxiv.org/abs/2402.03300)
