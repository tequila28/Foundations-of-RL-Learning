"""Read the original GSM8K JSONL files without a dataset library."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Example:
    question: str
    answer: str


def load_examples(path: str | Path, limit: int | None = None) -> list[Example]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    examples: list[Example] = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item.get("question"), str) or not isinstance(item.get("answer"), str):
                raise ValueError(f"{path}:{line_number} needs string question and answer fields")
            examples.append(Example(item["question"], item["answer"]))
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples
