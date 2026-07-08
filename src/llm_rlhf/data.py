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