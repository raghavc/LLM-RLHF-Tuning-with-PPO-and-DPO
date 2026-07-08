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