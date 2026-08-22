import pytest
import torch

from llm_rlhf.objectives import dpo_loss, group_relative_advantages, leave_one_out_advantages, ppo_clipped_policy_loss


def test_ppo_clipped_policy_loss_clips_large_update() -> None:
    new = torch.log(torch.tensor([2.0, 0.5]))
    old = torch.zeros(2)
    advantages = torch.tensor([1.0, -1.0])
    loss = ppo_clipped_policy_loss(new, old, advantages, clip_range=0.2)
    assert loss.item() == pytest.approx(-0.2)


def test_dpo_loss_rewards_better_policy_preference() -> None:
    good_loss, metrics = dpo_loss(
        torch.tensor([3.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        beta=1.0,
    )
    neutral_loss, _ = dpo_loss(