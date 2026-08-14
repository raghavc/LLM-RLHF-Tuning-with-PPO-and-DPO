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
    tokenizer: Any,
    num_proc: int | None,
    max_prompt_length: int,
) -> Dataset:
    def tokenize(example: dict[str, Any]) -> dict[str, Any]:
        input_ids = tokenizer(example["prompt"], padding=False)["input_ids"]
        return {"input_ids": input_ids, "length": len(input_ids)}

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names, num_proc=num_proc)
    return tokenized.filter(lambda example: example["length"] <= max_prompt_length, num_proc=num_proc)


def main() -> None:
    parser = TrlParser((DataArguments, PPOConfig, ModelConfig))
    data_args, training_args, model_args = parser.parse_args_and_config()
    train_dataset, eval_dataset = load_training_datasets(
        data_args,
        require_eval=wants_evaluation(training_args),
        schema="ppo",
    )
    if not isinstance(train_dataset, Dataset) or (eval_dataset is not None and not isinstance(eval_dataset, Dataset)):
        raise ValueError("PPO requires an in-memory dataset; disable streaming.")

    model_kwargs = _model_kwargs(model_args)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        num_labels=1,
        **model_kwargs,
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        num_labels=1,
        **model_kwargs,
    )
    policy = AutoModelForCausalLM.from_pretrained(training_args.sft_model_path, **model_kwargs)
    expected_vocab_size = len(tokenizer)
    for name, model in (("policy", policy), ("reward", reward_model), ("value", value_model)):
        if model.config.vocab_size != expected_vocab_size:
            raise ValueError(
                f"The {name} model vocabulary ({model.config.vocab_size}) does not match the tokenizer "
                f"({expected_vocab_size}). PPO requires one shared token vocabulary."
            )
    peft_config = get_peft_config(model_args)
    ref_policy = None
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(training_args.sft_model_path, **model_kwargs)

    max_positions = getattr(policy.config, "max_position_embeddings", 2048)
    max_prompt_length = min(1024, max_positions - training_args.response_length)
    if max_prompt_length < 1:
        raise ValueError("response_length must be smaller than the policy context window.")

    with PartialState().local_main_process_first():
        train_dataset = _tokenize_prompts(
            train_dataset,
            tokenizer,
            training_args.dataset_num_proc,
            max_prompt_length,
        )
        if eval_dataset is not None:
            eval_dataset = _tokenize_prompts(
                eval_dataset,
                tokenizer,
                training_args.dataset_num_proc,
                max_prompt_length,
            )

    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"Training complete. Model saved to {training_args.output_dir}.")
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=data_args.dataset_name)


if __name__ == "__main__":
    main()
