# LLM-RLHF-Tuning

This project implements the major stages of language-model post-training with clear configurations, small reference objectives, and current Hugging Face Transformers and TRL APIs. It includes supervised fine-tuning, reward modeling, PPO, direct preference optimization, binary-feedback alignment, and online reinforcement learning for verifiable tasks.

## Version 2.0

Version 2.0 replaces the 2023-era training stack with Transformers 5, TRL 1, PEFT, Accelerate, and Qwen3 examples. It also corrects the original documentation: DPO in this project means **Direct Preference Optimization**, not Deterministic Policy Optimization.

The release adds:

- **Supervised Fine-Tuning**: Conversational and prompt-completion datasets, sequence packing, LoRA, rank-stabilized LoRA, and optional QLoRA.
- **Reward Model Training**: Pairwise Bradley-Terry training with optional reward centering.
- **PPO Training**: End-to-end policy, reference, reward, and value-model training through the current TRL experimental PPO trainer.
- **Direct Preference Optimization**: DPO plus the loss families exposed by TRL, including IPO, robust DPO, EXO, NCA, AOT, APO, and DiscoPOP.
- **KTO Training**: Alignment from unpaired desirable and undesirable completions.
- **GRPO-Family Training**: GRPO, DAPO, Dr. GRPO, CISPO, SAPO, LUSPO, and VESPO objectives through one configurable online trainer.
- **RLOO Training**: A simpler REINFORCE leave-one-out alternative to PPO for online RLHF.
- **Verifiable Rewards**: Exact-answer, structured-format, and bounded length reward functions for reasoning experiments.
- **Reproducible Tooling**: Versioned YAML recipes, schema checks, unit tests, linting, and continuous integration.

## Installation

Python 3.10 or newer is required. The pinned environment used for release 2.0 is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For 4-bit QLoRA training:

```bash
pip install -e '.[quantization]'
```

The base requirements pin PyTorch 2.13.0, Transformers 5.14.1, TRL 1.9.0, PEFT 0.20.0, Accelerate 1.14.0, and Datasets 5.0.1. Install the PyTorch build appropriate for your CUDA environment when the default wheel is not suitable.

## Quick Start

Every trainer accepts a YAML file and command-line overrides.

```bash
python -m llm_rlhf.sft --config configs/sft.yaml
python -m llm_rlhf.reward --config configs/reward.yaml
python -m llm_rlhf.dpo --config configs/dpo.yaml
python -m llm_rlhf.kto --config configs/kto.yaml
python -m llm_rlhf.grpo --config configs/grpo.yaml
python -m llm_rlhf.rloo --config configs/rloo.yaml
```

PPO needs a reward model that uses the same tokenizer and vocabulary as the policy. Run the reward-model stage first, then:

```bash
python -m llm_rlhf.ppo --config configs/ppo.yaml
```

For distributed training:

```bash
accelerate launch -m llm_rlhf.dpo --config configs/dpo.yaml
```

For DeepSpeed ZeRO-3, edit `num_processes` in the supplied Accelerate configuration to match the machine:

```bash
accelerate launch \
  --config_file configs/accelerate/deepspeed_zero3.yaml \
  -m llm_rlhf.grpo \
  --config configs/grpo.yaml
```

The recipes target a recent NVIDIA GPU with bfloat16 support. For a CPU parser or smoke run, override `--bf16 false --dtype float32` and reduce the batch, sequence, and generation settings.

## Mathematical Foundations

### Proximal Policy Optimization

PPO constrains policy updates with a clipped likelihood ratio:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\right)\right]
$$

where $r_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\theta_{\text{old}}}(a_t\mid s_t)$ and $\hat{A}_t$ is an advantage estimate. The complete trainer is in `llm_rlhf.ppo`; the small, inspectable loss is in `llm_rlhf.objectives`.

### Direct Preference Optimization

DPO optimizes a policy from preferred and rejected completions without fitting a separate reward model:

$$
L_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right]\right)
$$

The reference policy regularizes the update, while $\beta$ controls the strength of that regularization.

### Group Relative Policy Optimization

GRPO samples a group of completions for each prompt and forms relative advantages without a learned value model:

$$
\hat{A}_i = \frac{r_i-\mathrm{mean}(r_1,\ldots,r_G)}{\mathrm{std}(r_1,\ldots,r_G)+\epsilon}
$$

The 2.0 GRPO recipe uses the DAPO token-level loss, asymmetric clipping, truncated-completion masking, and no group standard-deviation scaling. These choices address length and prompt-difficulty biases identified after the original GRPO formulation.

### REINFORCE Leave-One-Out

RLOO replaces a value model with the mean reward of the other samples in the same prompt group:

$$
b_i = \frac{1}{G-1}\sum_{j\ne i}r_j, \qquad \hat{A}_i=r_i-b_i
$$

This makes RLOO a useful modern comparison to PPO when online sampling is available.

## Supported Models

The release recipes use **Qwen3-0.6B** so that the complete pipeline can be prototyped on modest hardware. The trainers use Transformers Auto classes and can be configured for other compatible text models, including larger Qwen3 checkpoints, Gemma 3, Llama 3.x, and Mistral-family checkpoints.

Qwen3.5 is documented in the research notes as a current architecture, but it is natively multimodal. Use a multimodal dataset and verify that the selected trainer supports its processor and model class before substituting it into a text-only recipe.

## Supported Training Methods

**Full Fine-Tuning**

**LoRA and Rank-Stabilized LoRA**

**QLoRA with NF4 Quantization**

**Single-GPU, Accelerate, and DeepSpeed ZeRO-3 Training**

## Data Formats

The loader accepts Hub datasets or local JSON, JSONL, CSV, TSV, Parquet, and text files. It validates each trainer's required columns before model loading. Minimal examples are under `data/examples`; complete schemas and conversion notes are in [docs/DATASETS.md](docs/DATASETS.md).

## Research Notes

[docs/RESEARCH.md](docs/RESEARCH.md) explains which method to use, what the current configuration options mean, and which primary papers motivated the 2.0 implementation. [docs/MIGRATION.md](docs/MIGRATION.md) maps the previous scripts and flags to the new entry points.

## Updates

- **[2026-08-14]** Released version 2.0 with current Transformers and TRL APIs, Qwen3 recipes, PPO, DPO, KTO, GRPO/DAPO, RLOO, verifiable rewards, tests, and research documentation.
- **[2024-02-07]** Added LLaMA 2, DPO, single-base-model PPO, LoRA adapter configurations, Accelerate, and DeepSpeed support.
- **[2024-03-05]** Added LLaMA and two-base-model PPO training with Accelerate distributed support.

## Validation

```bash
pip install -e '.[dev]'
ruff check .
pytest
```

Training quality still depends on dataset quality, reward validity, evaluation design, and hardware-specific hyperparameters. The example recipes are research starting points, not claims of benchmark reproduction.
