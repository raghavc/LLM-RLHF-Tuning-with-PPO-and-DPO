"""Shared trainer setup and save helpers."""

from typing import Any


def wants_evaluation(training_args: Any) -> bool:
    strategy = getattr(training_args, "eval_strategy", "no")
    return getattr(strategy, "value", strategy) != "no"


def configure_model_loading(training_args: Any, model_args: Any) -> None:
    training_args.model_init_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": training_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "dtype": model_args.dtype,
    }


def save_trainer(trainer: Any, output_dir: str, dataset_name: str | None) -> None:
    trainer.save_model(output_dir)
    trainer.accelerator.print(f"Training complete. Model saved to {output_dir}.")
    if trainer.args.push_to_hub:
        trainer.push_to_hub(dataset_name=dataset_name)
