"""Evaluate exact final-answer accuracy on the GSM8K test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import load_examples
from .modeling import DEFAULT_MODEL, completion_mask, decode_responses, default_device, prompt_ids
from .reward import correctness_reward


CHAPTER_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-jsonl", type=Path, default=CHAPTER_ROOT / "data/test.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model or saved PPO/GRPO directory")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--predictions", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(args.test_jsonl, args.max_examples)
    device = default_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    correct = 0
    records = []
    for index, example in enumerate(examples, 1):
        prompt = prompt_ids(tokenizer, example.question, device)
        with torch.inference_mode():
            sequences = model.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        mask = completion_mask(sequences[:, prompt.shape[1]:], tokenizer.eos_token_id)
        response = decode_responses(tokenizer, sequences, mask, prompt.shape[1])[0]
        reward = correctness_reward(response, example.answer)
        correct += int(reward)
        records.append({
            "question": example.question,
            "gold_answer": example.answer,
            "response": response,
            "correct": bool(reward),
        })
        print(f"{index}/{len(examples)} correct={correct} accuracy={correct / index:.3f}")
    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Final exact-answer accuracy: {correct}/{len(examples)} = {correct / len(examples):.3f}")


if __name__ == "__main__":
    main()
