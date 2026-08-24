import pytest

from llm_rlhf.rewards import (
    completion_text,
    exact_answer_reward,
    extract_final_answer,
    format_reward,
    normalize_answer,
    resolve_reward_functions,
    soft_length_reward,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<answer>42</answer>", "42"),
        ("The result is \\boxed{42}", "42"),
        ("work\n#### 42", "42"),
        ("work\n42", "42"),
    ],
)
def test_extract_final_answer(text: str, expected: str) -> None:
    assert extract_final_answer(text) == expected
