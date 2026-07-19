"""End-to-end PPO training through TRL's current experimental trainer."""

from typing import Any

import torch
from accelerate import PartialState
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import ModelConfig, TrlParser, get_peft_config, get_quantization_config
from trl.experimental.ppo import PPOConfig, PPOTrainer

from .arguments import DataArguments
from .data import load_training_datasets
from .training import wants_evaluation


def _model_kwargs(model_args: ModelConfig) -> dict[str, Any]:
    dtype = model_args.dtype if model_args.dtype in {"auto", None} else getattr(torch, model_args.dtype)
    kwargs: dict[str, Any] = {
        "revision": model_args.model_revision,
        "attn_implementation": model_args.attn_implementation,
        "dtype": dtype,
    }
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    return kwargs


def _tokenize_prompts(
    dataset: Dataset,