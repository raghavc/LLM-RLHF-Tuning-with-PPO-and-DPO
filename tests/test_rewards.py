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


def test_completion_text_supports_conversational_content() -> None:
    completion = [{"role": "assistant", "content": [{"type": "text", "text": "final"}]}]
    assert completion_text(completion) == "final"


def test_exact_answer_reward_normalizes_numeric_answers() -> None:
    completions = ["<answer>1,024.0</answer>", "<answer>7</answer>"]
    assert exact_answer_reward(completions, ["1024", "8"]) == [1.0, 0.0]
    assert normalize_answer("The answer is 42.") == "42"


def test_format_and_length_rewards_are_bounded() -> None:
    valid = "<think>brief work</think>\n<answer>42</answer>"
    assert format_reward([valid, "42"]) == [1.0, 0.0]
    assert soft_length_reward(["one two", "one two three four"], soft_length_limit=2, hard_length_limit=4) == [
        0.0,
        -1.0,
    ]


def test_resolve_reward_functions_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown reward"):
        resolve_reward_functions(["missing"])
