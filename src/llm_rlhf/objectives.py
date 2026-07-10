"""Small, inspectable implementations of the central optimization objectives."""

from typing import Any

import torch
from torch.nn import functional as F


def _mean(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    denominator = mask.sum()
    if denominator.item() == 0:
        raise ValueError("mask must select at least one element")
    return (values * mask).sum() / denominator


def ppo_clipped_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
    mask: torch.Tensor | None = None,
) -> torch.Tensor: