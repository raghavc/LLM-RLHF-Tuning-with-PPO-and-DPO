"""GRPO-family online reinforcement-learning entry point."""

from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser, get_peft_config, get_quantization_config

from .arguments import DataArguments, OnlineArguments
from .data import load_training_datasets
from .rewards import resolve_reward_functions
from .training import configure_model_loading, save_trainer, wants_evaluation


def main() -> None:
    parser = TrlParser((DataArguments, OnlineArguments, GRPOConfig, ModelConfig))
    data_args, online_args, training_args, model_args = parser.parse_args_and_config()
    configure_model_loading(training_args, model_args)
    train_dataset, eval_dataset = load_training_datasets(
        data_args,
        require_eval=wants_evaluation(training_args),
        schema="online",
    )
    rewards = resolve_reward_functions(
        online_args.reward_functions,
        online_args.reward_model_name_or_path,
    )
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=rewards,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        quantization_config=get_quantization_config(model_args),
        peft_config=get_peft_config(model_args),
    )
    trainer.train()
    save_trainer(trainer, training_args.output_dir, data_args.dataset_name)


if __name__ == "__main__":
    main()
