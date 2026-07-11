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
    """Compute the negative PPO clipped surrogate objective."""

    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    return -_mean(torch.minimum(unclipped, clipped), mask)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the Direct Preference Optimization logistic loss."""

    policy_log_ratios = policy_chosen_logps - policy_rejected_logps
    reference_log_ratios = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_log_ratios - reference_log_ratios)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    metrics = {
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,