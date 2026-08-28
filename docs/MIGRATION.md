# Migrating from Version 1

Version 2.0 is a deliberate API break. The previous code targeted Transformers 4.31, TRL 0.5, PEFT 0.4, and PyTorch 2.0. Those APIs cannot share one reliable environment with the new trainers.

## Entry Point Mapping

| Version 1 | Version 2 |
| --- | --- |
| `script/sft/run_sft_with_peft.py` | `python -m llm_rlhf.sft --config configs/sft.yaml` |
| `script/rm/run_rm_with_peft.py` | `python -m llm_rlhf.reward --config configs/reward.yaml` |
| `script/ppo/run_ppo_with_peft.py` | `python -m llm_rlhf.ppo --config configs/ppo.yaml` |
| `script/dpo/run_dpo_with_peft.py` | `python -m llm_rlhf.dpo --config configs/dpo.yaml` |
| No equivalent | `python -m llm_rlhf.kto --config configs/kto.yaml` |
| No equivalent | `python -m llm_rlhf.grpo --config configs/grpo.yaml` |
| No equivalent | `python -m llm_rlhf.rloo --config configs/rloo.yaml` |

## Configuration Changes
