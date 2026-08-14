"""Dataset loading and schema validation."""

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset

DatasetLike = Dataset | IterableDataset
DatasetCollection = DatasetDict | IterableDatasetDict


def _local_builder(path: str) -> tuple[str, dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return "json", {}
    if suffix == ".csv":
        return "csv", {}
    if suffix == ".tsv":
        return "csv", {"delimiter": "\t"}
    if suffix == ".parquet":
        return "parquet", {}
    if suffix in {".txt", ".text"}:
        return "text", {}
    raise ValueError(f"Unsupported local dataset extension: {suffix or '<none>'}")


def load_training_datasets(
    data_args: Any,
    *,
    require_eval: bool,
    schema: str,
) -> tuple[DatasetLike, DatasetLike | None]:
    """Load, normalize, split, and validate a trainer dataset."""

    if data_args.train_file:
        builder, builder_kwargs = _local_builder(data_args.train_file)
        data_files = {data_args.train_split: data_args.train_file}
        if data_args.eval_file:
            eval_builder, eval_kwargs = _local_builder(data_args.eval_file)
            if (eval_builder, eval_kwargs) != (builder, builder_kwargs):
                raise ValueError("train_file and eval_file must use the same format.")
            data_files[data_args.eval_split] = data_args.eval_file
        dataset = load_dataset(builder, data_files=data_files, streaming=data_args.streaming, **builder_kwargs)
    else:
        dataset = load_dataset(
            data_args.dataset_name,
            name=data_args.dataset_config,
            streaming=data_args.streaming,
        )

    dataset = _rename_columns(dataset, data_args.column_map)
    if data_args.train_split not in dataset:
        raise ValueError(f"Training split '{data_args.train_split}' was not found. Available splits: {list(dataset)}")

    train_dataset = dataset[data_args.train_split]
    eval_dataset = dataset.get(data_args.eval_split) if require_eval else None

    if require_eval and eval_dataset is None:
        if data_args.streaming:
            raise ValueError("Streaming datasets need an explicit evaluation split or eval_file.")
        split = train_dataset.train_test_split(test_size=data_args.test_size, seed=42)
        train_dataset, eval_dataset = split["train"], split["test"]

    validate_schema(train_dataset, schema)
    if eval_dataset is not None:
        validate_schema(eval_dataset, schema)
    return train_dataset, eval_dataset


def _rename_columns(dataset: DatasetCollection, column_map: dict[str, str]) -> DatasetCollection:
    if not column_map:
        return dataset
    missing = set(column_map) - set(next(iter(dataset.values())).column_names)
    if missing:
        raise ValueError(f"Columns requested by column_map were not found: {sorted(missing)}")
    return dataset.rename_columns(column_map)


def validate_schema(dataset: DatasetLike, schema: str) -> None:
    """Fail early with a useful error when a dataset does not match a trainer."""

    columns = set(dataset.column_names)
    accepted: dict[str, tuple[frozenset[str], ...]] = {
        "sft": (frozenset({"messages"}), frozenset({"text"}), frozenset({"prompt", "completion"})),
        "preference": (frozenset({"chosen", "rejected"}),),
        "kto": (frozenset({"prompt", "completion", "label"}),),
        "online": (frozenset({"prompt"}),),
        "ppo": (frozenset({"prompt"}),),
    }
    if schema not in accepted:
        raise ValueError(f"Unknown dataset schema: {schema}")
    if any(required <= columns for required in accepted[schema]):
        return
    expected = " or ".join("{" + ", ".join(sorted(group)) + "}" for group in accepted[schema])
    raise ValueError(f"Dataset for {schema} must contain {expected}; found {sorted(columns)}")
