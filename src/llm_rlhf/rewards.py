"""Deterministic reward functions for verifiable online RL tasks."""

import importlib
import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

RewardFunction = Callable[..., list[float]]

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_GSM8K = re.compile(r"####\s*([^\n]+)")
_STRICT_FORMAT = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)


def completion_text(completion: Any) -> str:
    """Return text from standard or conversational TRL completions."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return _content_text(completion.get("content", ""))
    if isinstance(completion, list) and completion:
        return completion_text(completion[-1])
    return str(completion)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
        return "".join(parts)
    return str(content)


def extract_final_answer(text: str) -> str:
    """Extract an answer from common reasoning-dataset conventions."""

    for pattern in (_ANSWER_TAG, _BOXED, _GSM8K):
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def normalize_answer(answer: Any) -> str:
    text = str(answer).strip().lower().replace(",", "")
    text = re.sub(r"^(the answer is|answer:)\s*", "", text)
    text = text.strip(" .$`\n\t")
    try: