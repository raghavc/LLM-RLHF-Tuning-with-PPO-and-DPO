# LLM-RLHF-Tuning

This project implements Reinforcement Learning from Human Feedback (RLHF) training from the ground up. It includes detailed documentation of the implementation process and welcomes community discussions and contributions.


## Main Features

- **Instruction Fine-Tuning**: Support for fine-tuning the Alpaca model using specific instructions.
- **Reward Model Training**: Includes functionality to train a reward model effectively.
- **PPO Algorithm Training**: Offers comprehensive support for training RL models using the Proximal Policy Optimization (PPO) algorithm with various configurations:
    - Two base models with two LoRA adapters, supporting accelerate distributed training.
    - A single base model with two LoRA adapters, supporting accelerate and deepspeed training.
    - A single base model with one LoRA adapter, where Actor and Critic share the base model, also supporting accelerate and deepspeed training.
- **DPO Algorithm Training**: Support for training models using the DPO algorithm.

## Updates

- **[02/7/2024]** Added support for training LLaMA2 models and DPO training. Introduced PPO training based on a single base model, with an option for one or two LoRA adapters, and included support for accelerate and deepspeed training.
- **[03/5/13]** Introduced support for LLaMA model training and PPO training based on two base models with two LoRA adapters, along with accelerate distributed training.

## Mathematical Foundations

### Proximal Policy Optimization (PPO)

PPO is an optimization algorithm used in reinforcement learning to update policy parameters by optimizing a clipped surrogate objective function. The objective function $L^{CLIP}(\theta)$ for PPO is defined as:

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \cdot \hat{A}_t, \text{clip} \left( r_t(\theta), 1 - \epsilon, 1 + \epsilon \right) \cdot \hat{A}_t \right) \right]
$$

Where:
$$r_t(\theta)$$
- is the ratio of the probability of taking an action under the new policy to that under the old policy.
$$\hat{A}_t$$ 
- is the estimated advantage function at time step `t`.
$$\epsilon $$ 
- is a hyperparameter representing the clip range.

### Deterministic Policy Optimization (DPO)

DPO is another reinforcement learning algorithm that directly optimizes the deterministic policy to maximize the expected return. It updates the policy parameters $\theta$ by maximizing the expected return $J(\theta)$, given by:

$$
J(\theta) = \int_{\mathcal{S}} \rho^{\pi}(s) \int_{\mathcal{A}} \pi(s, a; \theta) Q^{\pi}(s, a) \, da \, ds
$$

Where:
- $\rho^{\pi}$ is the state-visitation distribution under policy $\pi$.
- $Q^{\pi}(s, a)$ is the state-action value function.



To set up your environment for the project, ensure you have the following dependencies installed:

```bash
accelerate==0.21.0
datasets==2.13.1
scikit-learn==1.3.0
sentencepiece==0.1.99
tqdm==4.65.0
transformers==4.31.0
wandb==0.15.8
peft==0.4.0
torch==2.0.1
trl==0.5.0
deepspeed==0.10.0
```

## Supported Models
**LLaMA**

**LLaMA2**
## Supported Training Methods
**LoRA**
