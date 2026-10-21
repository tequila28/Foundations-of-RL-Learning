"""Minimal token-level PPO with a value head and GAE for GSM8K."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch
from torch import nn

from .data import load_examples
from .modeling import (
    DEFAULT_MODEL, decode_responses, default_device, load_policy_and_reference,
    masked_mean, prompt_ids, sample_group, save_policy, score_completions,
    seed_everything,
)
from .reward import correctness_reward


CHAPTER_ROOT = Path(__file__).resolve().parents[1]


def gae_advantages(rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor,
                   gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE over generated tokens; the answer reward is on the final token."""
    advantages = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0], device=rewards.device)
    for position in range(rewards.shape[1] - 1, -1, -1):
        if position + 1 < rewards.shape[1]:
            next_mask = mask[:, position + 1].float()
            next_value = values[:, position + 1]
        else:
            next_mask = torch.zeros_like(running)
            next_value = torch.zeros_like(running)
        delta = rewards[:, position] + gamma * next_value * next_mask - values[:, position]
        running = (delta + gamma * gae_lambda * next_mask * running) * mask[:, position]
        advantages[:, position] = running
    return advantages, advantages + values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=CHAPTER_ROOT / "data/train.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=CHAPTER_ROOT / "outputs/ppo")
    parser.add_argument("--steps", type=int, default=20, help="Number of prompt/update cycles")
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.ppo_epochs < 1:
        raise ValueError("steps and ppo-epochs must be positive")
    seed_everything(args.seed)
    examples = load_examples(args.train_jsonl, args.max_examples)
    random.Random(args.seed).shuffle(examples)
    device = default_device()
    tokenizer, policy, reference = load_policy_and_reference(args.model, device)
    value_head = nn.Linear(policy.config.hidden_size, 1).to(device)
    parameters = list(policy.parameters()) + list(value_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)

    for step in range(args.steps):
        example = examples[step % len(examples)]
        prompt = prompt_ids(tokenizer, example.question, device)
        sequences, mask = sample_group(
            policy, tokenizer, prompt, group_size=1,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )
        response = decode_responses(tokenizer, sequences, mask, prompt.shape[1])[0]
        score = correctness_reward(response, example.answer)

        with torch.no_grad():
            old_logp, old_values = score_completions(
                policy, sequences, prompt.shape[1], mask, value_head
            )
            reference_logp, _ = score_completions(
                reference, sequences, prompt.shape[1], mask
            )
            # Sampled KL cost on each action, followed by the terminal task reward.
            rewards = -args.kl_coef * (old_logp - reference_logp) * mask
            last_token = mask.long().sum(dim=1) - 1
            rewards[torch.arange(mask.shape[0], device=device), last_token] += score
            advantages, returns = gae_advantages(
                rewards, old_values, mask, args.gamma, args.gae_lambda
            )

        # Eval mode disables dropout but does not disable gradients.
        policy.eval()
        for _ in range(args.ppo_epochs):
            new_logp, new_values = score_completions(
                policy, sequences, prompt.shape[1], mask, value_head
            )
            ratio = (new_logp - old_logp).clamp(-20, 20).exp()
            unclipped = ratio * advantages
            clipped = ratio.clamp(1 - args.clip_epsilon, 1 + args.clip_epsilon) * advantages
            policy_loss = -masked_mean(torch.minimum(unclipped, clipped), mask)
            value_loss = 0.5 * masked_mean((new_values - returns).square(), mask)
            loss = policy_loss + args.value_coef * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

        print(
            f"step={step + 1}/{args.steps} reward={score:.0f} "
            f"policy_loss={policy_loss.item():.4f} value_loss={value_loss.item():.4f}"
        )

    save_policy(policy, tokenizer, args.output_dir, value_head)
    print(f"Saved policy and value head to {args.output_dir}")


if __name__ == "__main__":
    main()
