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
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        beta=1.0,
    )
    assert good_loss < neutral_loss
    assert metrics["reward_accuracy"].item() == 1.0


def test_group_relative_advantages_are_centered() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    advantages = group_relative_advantages(rewards)
    assert advantages.mean().item() == pytest.approx(0.0, abs=1e-6)


def test_leave_one_out_advantages() -> None:
    rewards = torch.tensor([[1.0, 2.0, 4.0]])
    advantages = leave_one_out_advantages(rewards)
    assert advantages.tolist()[0] == pytest.approx([-2.0, -0.5, 2.5])
    with pytest.raises(ValueError, match="at least two"):
        leave_one_out_advantages(torch.tensor([[1.0]]))
