# Dataset Formats

Version 2.0 accepts either a Hugging Face dataset identifier or local files. Set exactly one of `dataset_name` and `train_file`. When the source has no evaluation split, `test_size` creates one unless streaming is enabled.

## Supervised Fine-Tuning

Use a conversational `messages` column:

```json
{"messages":[{"role":"user","content":"Explain KL divergence."},{"role":"assistant","content":"KL divergence measures how one probability distribution differs from another."}]}
```

TRL also accepts a `text` column or a `prompt` and `completion` pair. Conversational data is preferred for chat models because the tokenizer's chat template controls the exact wire format.

## Reward Modeling and DPO

Pairwise datasets require `chosen` and `rejected`. An explicit `prompt` is recommended:

```json
{"prompt":"What is 6 times 7?","chosen":"42","rejected":"36"}
```

The three values may also be conversational message lists. Reward modeling fits a scalar preference score. DPO consumes the same pair directly.

## KTO

KTO uses unpaired binary feedback:

```json
{"prompt":"What is 6 times 7?","completion":"42","label":true}
{"prompt":"What is 6 times 7?","completion":"36","label":false}
```

The `label` field must be boolean. Check the desirable-to-undesirable ratio and adjust `desirable_weight` or `undesirable_weight` when it is materially imbalanced.

## GRPO and RLOO

Online trainers require `prompt`. Built-in exact-answer rewards also require `answer`:

```json
{"prompt":"A box has 7 rows of 6 pencils. How many pencils are there?","answer":"42"}
```

The exact-answer extractor recognizes `<answer>...</answer>`, `\boxed{...}`, GSM8K's `#### ...` convention, and finally the last non-empty line. For new datasets, prefer an explicit answer boundary.

The format reward expects the full completion to use:

```text
<think>reasoning</think>
<answer>final answer</answer>
```

Only use a format reward when the prompt or chat template asks for this structure. Otherwise the reward will be uninformative.

## PPO

PPO requires a `prompt` column. Its reward checkpoint must share the policy tokenizer and vocabulary because the current trainer scores policy token IDs directly with the reward and value models.

## Column Mapping

Rename source columns in YAML without preprocessing a new copy:

```yaml
column_map:
  question: prompt
  solution: answer
```

Mapping runs before schema validation and applies to every split.

## Local Files

```yaml
train_file: data/examples/preference.jsonl
eval_file: null
test_size: 0.25
```

Supported extensions are `.json`, `.jsonl`, `.csv`, `.tsv`, `.parquet`, `.txt`, and `.text`. Train and evaluation files must use the same format.
