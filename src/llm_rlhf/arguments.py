"""Shared command-line argument definitions."""

from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """Dataset source, split, and column-normalization options."""

    dataset_name: str | None = field(
        default=None,
        metadata={"help": "Hugging Face dataset identifier or dataset builder name."},
    )
    dataset_config: str | None = field(
        default=None,
        metadata={"help": "Optional named configuration for a Hub dataset."},
    )
    train_file: str | None = field(
        default=None,
        metadata={"help": "Local JSON, JSONL, CSV, TSV, Parquet, or text training file."},