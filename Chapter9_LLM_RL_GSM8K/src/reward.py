"""A simple GSM8K rule reward: compare the final number after ####."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re


FINAL_ANSWER = re.compile(r"####\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")


def final_number(text: str) -> Decimal | None:
    matches = FINAL_ANSWER.findall(text)
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None


def correctness_reward(response: str, gold_answer: str) -> float:
    gold = final_number(gold_answer)
    if gold is None:
        raise ValueError("GSM8K gold answer has no final number after ####")
    predicted = final_number(response)
    return float(predicted is not None and predicted == gold)
