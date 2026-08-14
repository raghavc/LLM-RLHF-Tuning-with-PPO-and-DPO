# Research Notes

Version 2.0 treats post-training as a set of related experimental choices rather than one fixed RLHF pipeline. The code exposes stable trainer entry points while the YAML files keep the algorithmic assumptions visible.

## Method Selection

| Available feedback | Recommended starting point | Main tradeoff |
| --- | --- | --- |
| Demonstrations | SFT | Simple and stable, but does not model relative preference |
| Chosen and rejected pairs | DPO | Offline and efficient, but depends on the data-generating policy and reference choice |
| Individual good or bad labels | KTO | Avoids paired collection, but needs calibrated class weights |
| Pairwise labels plus a learned scorer | Reward model, then PPO | Fully online optimization, but uses policy, reference, reward, and value models |
| Verifiable scalar rewards | GRPO-family or RLOO | Avoids a learned reward model, but is sensitive to reward design and sampling cost |

## PPO and RLOO

PPO remains an important RLHF baseline because it clips large policy updates and learns a value function. It also carries substantial memory and tuning cost. RLOO uses the other sampled completions for a leave-one-out baseline and removes the learned value model. The project includes both so comparisons can hold the policy, reward, and dataset constant.

Primary sources:

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)

## Direct and Binary Preference Optimization

DPO converts the KL-regularized reward objective into a binary classification loss over preferred and rejected completions. KTO instead learns from desirable or undesirable completions one at a time. TRL 1.9 also exposes preference-loss variants through `DPOConfig.loss_type`, including IPO, robust DPO, EXO, NCA, AOT, APO, and DiscoPOP. Change one objective at a time and keep the effective batch, reference model, data, and evaluation fixed.

Primary sources:

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)

## GRPO, DAPO, and Later Objectives

GRPO removes the value model by comparing multiple completions sampled for the same prompt. Subsequent work identified two important biases:

- Per-sequence token averaging can create a response-length bias.
- Dividing by each group's reward standard deviation can overweight prompts according to within-group difficulty.

The default `configs/grpo.yaml` therefore uses the DAPO token aggregation, asymmetric clipping, truncated-completion masking, and `scale_rewards: none`. This combines explicit choices from DAPO and the Dr. GRPO analysis. They are hypotheses to evaluate, not universally optimal defaults.

TRL 1.9 also exposes CISPO, SAPO, LUSPO, and VESPO through `loss_type`. These newer objectives mainly change clipping, importance weighting, and token- or sequence-level aggregation. Dedicated recipes are intentionally omitted until the user selects an off-policy regime and evaluation design that makes the comparison meaningful.

Primary sources:

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
- [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)
- [LUSPO: Length-Unbiased Sequence Policy Optimization](https://arxiv.org/abs/2602.05261)
- [VESPO: Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training](https://arxiv.org/abs/2602.10693)

## Parameter-Efficient Training

All recipes use LoRA with rank-stabilized scaling. Set `use_rslora: false` for conventional LoRA, `use_dora: true` for DoRA, or `load_in_4bit: true` for QLoRA. Quantization changes memory and numerical behavior, so it should be reported as part of the experimental method.

Primary sources:

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- [A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA](https://arxiv.org/abs/2312.03732)

## Model Choice

Qwen3-0.6B is the default because it is a modern multilingual dense Transformer small enough for end-to-end experimentation. Qwen3.5 adds hybrid linear and full attention and native multimodality, which changes the processor, input schema, and often the trainer memory profile. It should not be treated as a drop-in text-only checkpoint without validation.

- [Qwen3 model documentation](https://huggingface.co/docs/transformers/model_doc/qwen3)
- [Qwen3.5 model documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)

## Evaluation Discipline

- Establish the base and SFT model scores before online RL.
- Keep a held-out set that never contributes to reward construction.
- Log reward components separately to detect compensation and reward hacking.
- Track completion length, truncation rate, KL, entropy, and exact task success together.
- Inspect sampled completions at fixed intervals; scalar reward alone is insufficient.
- Report model revision, dataset revision, prompt template, seed, precision, and dependency versions.
