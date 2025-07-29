from __future__ import annotations
from dataclasses import dataclass

from nn_model_args import NNModelArgs

"""
NN Model args for SimpleLoRA
"""

@dataclass
class NNModelArgs_SimpleLoRAMLP(NNModelArgs):

    input_dim: int
    hidden_dim: int
    output_dim: int
    rank_ratio: int = 1
    lora_mode: str = "standard"

    
