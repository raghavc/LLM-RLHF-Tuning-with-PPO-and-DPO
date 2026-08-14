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


def test_validate_schema_reports_missing_columns() -> None:
    with pytest.raises(ValueError, match="chosen"):
        validate_schema(Dataset.from_dict({"prompt": ["p"]}), "preference")


def test_local_loader_maps_columns_and_creates_eval_split(tmp_path) -> None:
    path = tmp_path / "data.jsonl"
    records = [{"question": f"q{i}", "solution": str(i)} for i in range(10)]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    args = DataArguments(
        train_file=str(path),
        test_size=0.2,
        column_map={"question": "prompt", "solution": "answer"},
    )

    train, evaluation = load_training_datasets(args, require_eval=True, schema="online")

    assert len(train) == 8
    assert evaluation is not None
    assert len(evaluation) == 2
    assert {"prompt", "answer"} <= set(train.column_names)
