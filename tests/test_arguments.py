import pytest

from llm_rlhf.arguments import DataArguments


def test_data_arguments_require_exactly_one_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        DataArguments()
    with pytest.raises(ValueError, match="exactly one"):
        DataArguments(dataset_name="org/data", train_file="train.jsonl")


def test_data_arguments_validate_eval_and_fraction() -> None:
    with pytest.raises(ValueError, match="eval_file"):
        DataArguments(dataset_name="org/data", eval_file="eval.jsonl")
    with pytest.raises(ValueError, match="test_size"):
        DataArguments(dataset_name="org/data", test_size=1.0)


def test_data_arguments_accept_hub_or_local_source() -> None:
    assert DataArguments(dataset_name="org/data").dataset_name == "org/data"
    assert DataArguments(train_file="train.jsonl").train_file == "train.jsonl"
