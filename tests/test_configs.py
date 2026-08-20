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
        ("reward.yaml", (DataArguments, RewardConfig, ModelConfig)),
        ("dpo.yaml", (DataArguments, DPOConfig, ModelConfig)),
        ("kto.yaml", (DataArguments, KTOConfig, ModelConfig)),
        ("grpo.yaml", (DataArguments, OnlineArguments, GRPOConfig, ModelConfig)),
        ("rloo.yaml", (DataArguments, OnlineArguments, RLOOConfig, ModelConfig)),
        ("ppo.yaml", (DataArguments, PPOConfig, ModelConfig)),
    ],
)
def test_release_configs_parse(filename: str, types: tuple[type, ...]) -> None:
    parser = TrlParser(types)
    parsed = parser.parse_args_and_config(
        args=["--config", str(ROOT / "configs" / filename), "--bf16", "false", "--dtype", "float32"]
    )
    assert len(parsed) == len(types)
