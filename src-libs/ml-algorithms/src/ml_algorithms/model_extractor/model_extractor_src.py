import sys
sys.path.insert(0, '')

import torch
import torch.nn as nn
import json
import os

from lora.lora_implementation.lora_linear import LoRALinear

class AdvancedModelExtractor:
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
        torch.save(flat, filepath if filepath.endswith(".npz") else filepath + ".npz")


# === ✅ 测试代码区域 ===
if __name__ == '__main__':
    import torch.nn.functional as F

    # 示例模型（CNN + LoRALinear）
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
            self.bn2 = nn.BatchNorm2d(32)
            self.fc1 = LoRALinear(32 * 5 * 5, 10, rank=2)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            return x

        # 示例模型（CNN + LoRALinear）
    class SimpleCNN_2(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
            self.bn2 = nn.BatchNorm2d(32)
            self.fc1 = LoRALinear(32 * 5 * 5, 10, rank=5)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            return x

    # 创建模型和提取器
    model = SimpleCNN()
    extractor = AdvancedModelExtractor(model)

    # ✅ 注册 BatchNorm 提取器
    def extract_batchnorm(module: nn.modules.batchnorm._BatchNorm):
        return {
            "weight": module.weight.detach().cpu(),
            "bias": module.bias.detach().cpu(),
            "running_mean": module.running_mean.detach().cpu(),
            "running_var": module.running_var.detach().cpu(),
            "num_batches_tracked": module.num_batches_tracked.item(),
            "layer_type": "batchnorm"
        }

    extractor.register_custom_handler(nn.BatchNorm2d, extract_batchnorm)

    # 输出结构化字典
    layer_dict = extractor.get_layer_dict()
    for layer_name, params in layer_dict.items():
        print(f"\nLayer: {layer_name}")
        for k, v in params.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {v}")

    extractor.model = SimpleCNN_2
    layer_dict_2 = extractor.get_layer_dict()

    print("\n✅ Extraction finished.")
