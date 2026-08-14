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

- `model_type` is removed. Transformers Auto classes select the architecture from the checkpoint.
- `load_in_4bit` remains available through `ModelConfig`; install the `quantization` extra first.
- `lora_rank` becomes `lora_r`.
- `lora_target` becomes `lora_target_modules`. The recipes use `all-linear` to avoid LLaMA-specific module lists.
- `dpo_beta` becomes `beta`.
- `max_prompt_length` and `max_response_length` become the trainer-specific `max_length` or `max_completion_length` settings.
- Dataset templates are replaced by tokenizer chat templates and canonical dataset schemas.
- `wandb` is optional. Set `report_to: wandb` and install the `tracking` extra when needed.

## PPO Changes

PPO still has an end-to-end trainer, but it is an experimental API in TRL 1.9. The policy, reference, reward, and value models are now explicit, and the recipe uses the lower-variance `k3` KL estimator. For new online-RL experiments, compare PPO against RLOO and GRPO-family objectives rather than treating PPO as the only baseline.

## Reproducing Version 1

The old implementation remains available in Git history before the 2.0 commit. Use a separate environment with the original `requirements.txt`; do not mix the version 1 and version 2 dependencies.
