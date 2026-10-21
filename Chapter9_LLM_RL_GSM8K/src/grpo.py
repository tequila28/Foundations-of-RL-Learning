"""Minimal GRPO: group-relative rewards, clipped policy loss, no critic."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import torch

from .data import load_examples
from .modeling import (
    DEFAULT_MODEL, decode_responses, default_device, load_policy_and_reference,
    prompt_ids, sample_group, save_policy, score_completions, seed_everything,
)
from .reward import correctness_reward


CHAPTER_ROOT = Path(__file__).resolve().parents[1]


def group_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Compare completions sampled for the same question."""
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("GRPO requires at least two rewards from one question")
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, default=CHAPTER_ROOT / "data/train.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=CHAPTER_ROOT / "outputs/grpo")
    parser.add_argument("--steps", type=int, default=20, help="Number of question/group updates")
    parser.add_argument("--max-examples", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--grpo-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.grpo_epochs < 1 or args.group_size < 2:
        raise ValueError("steps and grpo-epochs must be positive; group-size must be at least 2")
    seed_everything(args.seed)
    examples = load_examples(args.train_jsonl, args.max_examples)
    random.Random(args.seed).shuffle(examples)
    device = default_device()
    tokenizer, policy, reference = load_policy_and_reference(args.model, device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)

    for step in range(args.steps):
        example = examples[step % len(examples)]
        prompt = prompt_ids(tokenizer, example.question, device)
        sequences, mask = sample_group(
            policy, tokenizer, prompt, group_size=args.group_size,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )
        responses = decode_responses(tokenizer, sequences, mask, prompt.shape[1])
        rewards = torch.tensor(
            [correctness_reward(response, example.answer) for response in responses],
            dtype=torch.float32, device=device,
        )
        advantages = group_advantages(rewards)

        with torch.no_grad():
            old_logp, _ = score_completions(policy, sequences, prompt.shape[1], mask)
            reference_logp, _ = score_completions(reference, sequences, prompt.shape[1], mask)

        policy.eval()
        for _ in range(args.grpo_epochs):
            new_logp, _ = score_completions(policy, sequences, prompt.shape[1], mask)
            ratio = (new_logp - old_logp).clamp(-20, 20).exp()
            unclip = ratio * advantages[:, None]
            clipped = ratio.clamp(1 - args.clip_epsilon, 1 + args.clip_epsilon) * advantages[:, None]
            # Non-negative sampled KL estimate used in the GRPO objective.
            delta = (reference_logp - new_logp).clamp(-20, 20)
            kl = delta.exp() - delta - 1
            token_loss = -torch.minimum(unclip, clipped) + args.kl_coef * kl
            # Average tokens within each answer, then average answers in the group.
            loss = ((token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        print(
            f"step={step + 1}/{args.steps} group_reward={rewards.mean().item():.3f} "
            f"group_std={rewards.std(unbiased=False).item():.3f} loss={loss.item():.4f}"
        )

    save_policy(policy, tokenizer, args.output_dir)
    print(f"Saved policy to {args.output_dir}")


if __name__ == "__main__":
    main()
