"""REINFORCE Leave-One-Out online reinforcement-learning entry point."""

from trl import ModelConfig, RLOOConfig, RLOOTrainer, TrlParser, get_peft_config, get_quantization_config

from .arguments import DataArguments, OnlineArguments
from .data import load_training_datasets
from .rewards import resolve_reward_functions
from .training import configure_model_loading, save_trainer, wants_evaluation


def main() -> None:
    parser = TrlParser((DataArguments, OnlineArguments, RLOOConfig, ModelConfig))
    data_args, online_args, training_args, model_args = parser.parse_args_and_config()
    configure_model_loading(training_args, model_args)
    train_dataset, eval_dataset = load_training_datasets(
        data_args,
        require_eval=wants_evaluation(training_args),
        schema="online",
    )
    rewards = resolve_reward_functions(