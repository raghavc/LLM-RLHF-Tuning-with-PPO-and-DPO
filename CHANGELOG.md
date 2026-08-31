# Changelog

All notable changes to this project are documented in this file.

## 2.0.0 - 2026-08-14

### Added

- Current SFT, reward modeling, PPO, DPO, KTO, GRPO-family, and RLOO entry points.
- Qwen3 LoRA and rank-stabilized LoRA recipes for every training stage.
- DAPO and Dr. GRPO controls, asymmetric clipping, and truncated-completion masking.
- Exact-answer, structured-format, and bounded length rewards for verifiable tasks.
- Local and Hub dataset loading with split creation, column mapping, and schema validation.
- Tested reference implementations of PPO, DPO, GRPO, and RLOO objective components.
- Python packaging, CLI entry points, unit tests, linting, and GitHub Actions validation.
- Research, dataset, and migration documentation.

### Changed

- Upgraded the stack from Transformers 4.31 and TRL 0.5 to Transformers 5.14 and TRL 1.9.
- Replaced model-specific LLaMA classes with AutoModel and AutoTokenizer paths.
- Replaced custom argument parsing and trainers with maintained TRL configurations and trainers.
- Corrected DPO documentation to describe Direct Preference Optimization.

### Removed

- Obsolete version 1 trainer utilities, shell wrappers, and duplicated DeepSpeed configuration.
