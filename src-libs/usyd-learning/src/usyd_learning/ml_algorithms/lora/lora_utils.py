import torch.nn as nn
from .impl.lora_linear import LoRALinear

class LoRAUtils():
    @staticmethod
    def set_lora_mode_for_model(model: nn.Module, mode: str):
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.set_lora_mode(mode)
