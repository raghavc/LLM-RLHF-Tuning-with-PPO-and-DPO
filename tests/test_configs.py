from pathlib import Path

import pytest
from trl import DPOConfig, GRPOConfig, KTOConfig, ModelConfig, RewardConfig, RLOOConfig, SFTConfig, TrlParser
from trl.experimental.ppo import PPOConfig

from llm_rlhf.arguments import DataArguments, OnlineArguments

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "types"),
    [
        ("sft.yaml", (DataArguments, SFTConfig, ModelConfig)),