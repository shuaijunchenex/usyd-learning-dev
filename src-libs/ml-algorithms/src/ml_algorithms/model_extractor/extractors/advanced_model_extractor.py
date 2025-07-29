import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn
import json
import os

from lora.lora_linear import LoRALinear
from model_extractor.model_extractor_abc import BaseModelExtractor

class AdvancedModelExtractor(BaseModelExtractor):
    def __init__(self, model: nn.Module, only_trainable: bool = False):
        """
        Generalized model extractor for LoRA and standard layers with custom extractor support.

        Args:
            model (nn.Module): PyTorch model to extract parameters from.
            only_trainable (bool): Whether to include only trainable parameters.
        """
        self.model = model
        self.only_trainable = only_trainable
        self.custom_handlers = {}

        # 注册默认 LoRA 提取器
        self.register_custom_handler(LoRALinear, self._extract_lora_linear)

        # 初始提取
        self.layer_data = self._extract_all_layers()

    def register_custom_handler(self, layer_type: type, handler_fn):
        self.custom_handlers[layer_type] = handler_fn
        self.layer_data = self._extract_all_layers()  # refresh

    def set_model(self, model: nn.Module):
        """
        Set a new model and re-extract parameters.

        Args:
            model (nn.Module): New PyTorch model to extract parameters from.
        """
        self.model = model
        self.layer_data = self._extract_all_layers()

    def _extract_lora_linear(self, module):
        return {
            "weight": module.weight.detach().cpu(),
            "bias": module.bias.detach().cpu() if module.bias is not None else None,
            "lora_A": module.lora_A.detach().cpu(),
            "lora_B": module.lora_B.detach().cpu(),
            "layer_type": "lora"
        }

    def _extract_all_layers(self):
        data = {}
        for name, module in self.model.named_modules():
            if len(list(module.parameters(recurse=False))) == 0:
                continue

            extracted = None

            for custom_type, handler_fn in self.custom_handlers.items():
                if isinstance(module, custom_type):
                    extracted = handler_fn(module)
                    break

            if extracted is None:
                extracted = {
                    k: v.detach().cpu()
                    for k, v in module.named_parameters(recurse=False)
                    if not self.only_trainable or v.requires_grad
                }
                # 添加默认类型字段
                if extracted:
                    extracted["layer_type"] = type(module).__name__.lower()

            if extracted:
                data[name] = extracted

        return data

    def get_layer_dict(self):
        return self.layer_data

    def save_to_json(self, filepath: str):
        serializable = {
            layer: {
                name: tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor
                for name, tensor in params.items()
            }
            for layer, params in self.layer_data.items()
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2)

    def save_to_npz(self, filepath: str):
        flat = {}
        for layer, params in self.layer_data.items():
            for name, tensor in params.items():
                if isinstance(tensor, torch.Tensor):
                    key = f"{layer}.{name}"
                    flat[key] = tensor.numpy()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)