import json

import pytest
from datasets import Dataset

from llm_rlhf.arguments import DataArguments
from llm_rlhf.data import load_training_datasets, validate_schema


@pytest.mark.parametrize(
    ("schema", "columns"),
    [
        ("sft", {"messages": [[{"role": "user", "content": "Hi"}]]}),
        ("preference", {"chosen": ["yes"], "rejected": ["no"]}),
        ("kto", {"prompt": ["p"], "completion": ["c"], "label": [True]}),
        ("online", {"prompt": ["p"]}),
        ("ppo", {"prompt": ["p"]}),
    ],
)
def test_validate_schema_accepts_canonical_columns(schema: str, columns: dict) -> None:
    validate_schema(Dataset.from_dict(columns), schema)
