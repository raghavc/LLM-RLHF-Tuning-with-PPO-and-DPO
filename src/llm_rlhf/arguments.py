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
    )
    eval_file: str | None = field(
        default=None,
        metadata={"help": "Optional local evaluation file with the same format as train_file."},
    )
    train_split: str = field(default="train", metadata={"help": "Source split used for training."})
    eval_split: str = field(default="test", metadata={"help": "Source split used for evaluation."})
    test_size: float = field(
        default=0.02,
        metadata={"help": "Evaluation fraction created when the source has no evaluation split."},
    )
    column_map: dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Mapping from source column names to the canonical TRL schema."},
    )
    streaming: bool = field(default=False, metadata={"help": "Stream a Hub dataset instead of downloading it."})

    def __post_init__(self) -> None:
        if bool(self.dataset_name) == bool(self.train_file):
            raise ValueError("Set exactly one of dataset_name or train_file.")
        if self.eval_file and not self.train_file: