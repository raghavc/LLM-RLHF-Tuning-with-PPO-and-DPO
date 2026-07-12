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