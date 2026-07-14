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
        return format(Decimal(text).normalize(), "f")
    except InvalidOperation:
        return re.sub(r"\s+", " ", text)


def exact_answer_reward(completions: list[Any], answer: list[Any], **_: Any) -> list[float]:
    """Return one for an exact normalized answer and zero otherwise."""

    return [
        float(normalize_answer(extract_final_answer(completion_text(completion))) == normalize_answer(target))
        for completion, target in zip(completions, answer, strict=True)
    ]


def format_reward(completions: list[Any], **_: Any) -> list[float]:
    """Reward an explicit reasoning and final-answer boundary."""

    return [float(bool(_STRICT_FORMAT.fullmatch(completion_text(completion)))) for completion in completions]


def soft_length_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    """Apply no brevity bonus and a bounded penalty beyond a soft limit."""

    soft_limit = int(kwargs.get("soft_length_limit", 512))
    hard_limit = max(int(kwargs.get("hard_length_limit", 1024)), soft_limit + 1)
    rewards = []
    for completion in completions:
        length = len(completion_text(completion).split())
        overflow = max(length - soft_limit, 0)
        rewards.append(-min(overflow / (hard_limit - soft_limit), 1.0))
    return rewards


BUILTIN_REWARDS: dict[str, RewardFunction] = {
    "exact_answer": exact_answer_reward,
    "format": format_reward,
    "soft_length": soft_length_reward,
}


def resolve_reward_functions(names: list[str], reward_model: str | None = None) -> list[RewardFunction | str]:
    rewards: list[RewardFunction | str] = [reward_model] if reward_model else []
    for name in names:
        if name in BUILTIN_REWARDS:
            rewards.append(BUILTIN_REWARDS[name])
            continue
        if "." not in name:
            raise ValueError(f"Unknown reward function '{name}'. Built-ins: {sorted(BUILTIN_REWARDS)}")
        module_name, function_name = name.rsplit(".", 1)
        function = getattr(importlib.import_module(module_name), function_name)
        if not callable(function):
            raise TypeError(f"Reward target '{name}' is not callable.")
        rewards.append(function)
    if not rewards:
        raise ValueError("Configure at least one reward model or reward function.")
    return rewards
