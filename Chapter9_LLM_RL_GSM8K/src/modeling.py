"""Tokenizer, rollout and token-scoring helpers shared by PPO and GRPO."""

from __future__ import annotations

from pathlib import Path
import random

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "Solve the math problem step by step. End your response with exactly one "
    "line in the form #### <number>."
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_policy_and_reference(model_name: str, device: torch.device):
    """Keep a trainable policy and a frozen copy of its initial weights."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.eos_token_id is None:
        raise ValueError("This example requires an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    reference = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    reference.eval()
    reference.requires_grad_(False)
    return tokenizer, policy, reference


def prompt_ids(tokenizer, question: str, device: torch.device) -> torch.Tensor:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    return ids.to(device)


def completion_mask(completion_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Include the first EOS action but ignore EOS padding after it."""
    eos_hits = completion_ids.eq(eos_token_id)
    count = eos_hits.cumsum(dim=1)
    return (count == 0) | (eos_hits & (count == 1))


def sample_group(policy, tokenizer, prompt: torch.Tensor, group_size: int,
                 max_new_tokens: int, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    if group_size < 1 or max_new_tokens < 1 or temperature <= 0:
        raise ValueError("group_size, max_new_tokens and temperature must be positive")
    policy.eval()
    with torch.inference_mode():
        sequences = policy.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            do_sample=True,
            num_return_sequences=group_size,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Values produced in inference_mode cannot be saved for autograd during
    # the later policy update (embedding backward needs the token indices).
    sequences = sequences.clone()
    mask = completion_mask(sequences[:, prompt.shape[1]:], tokenizer.eos_token_id)
    return sequences, mask


def decode_responses(tokenizer, sequences: torch.Tensor, mask: torch.Tensor,
                     prompt_length: int) -> list[str]:
    completion_ids = sequences[:, prompt_length:]
    return [
        tokenizer.decode(tokens[valid], skip_special_tokens=True)
        for tokens, valid in zip(completion_ids, mask)
    ]


def score_completions(model, sequences: torch.Tensor, prompt_length: int,
                      mask: torch.Tensor, value_head: nn.Module | None = None
                      ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Align each completion token with the preceding causal LM prediction."""
    if prompt_length < 1 or sequences.shape[1] <= prompt_length:
        raise ValueError("Sequence must contain a prompt and at least one completion token")
    attention_mask = torch.cat(
        (torch.ones(sequences.shape[0], prompt_length, dtype=torch.long, device=sequences.device),
         mask.long()), dim=1
    )
    output = model(input_ids=sequences, attention_mask=attention_mask,
                   output_hidden_states=value_head is not None, use_cache=False,
                   return_dict=True)
    logits = output.logits[:, prompt_length - 1:-1, :].float()
    actions = sequences[:, prompt_length:]
    log_probs = torch.log_softmax(logits, dim=-1).gather(
        dim=-1, index=actions.unsqueeze(-1)
    ).squeeze(-1)
    values = None
    if value_head is not None:
        # Hidden state immediately before each generated action.
        states = output.hidden_states[-1][:, prompt_length - 1:-1]
        values = value_head(states).squeeze(-1).float()
    return log_probs, values


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def save_policy(policy, tokenizer, output_dir: str | Path,
                value_head: nn.Module | None = None) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(target)
    tokenizer.save_pretrained(target)
    if value_head is not None:
        torch.save(value_head.state_dict(), target / "value_head.pt")
