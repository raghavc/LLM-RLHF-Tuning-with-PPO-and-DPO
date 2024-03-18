import os,sys
import torch 
import torch.nn as nn 

PROMPT_TEMPLATE = dict(
    llama_alpaca=(
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: "
        ),
    llama2_alpaca=(
        "[INST] <<SYS>>\n"
        "You are a helpful assistant.\n"
        "<</SYS>>\n\n{instruction} [/INST]"
    ),
    default=(
        "Human: {instruction}\nAssistant: "
    )
)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)
























