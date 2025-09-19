from typing import Optional
import torch
import torch.nn as nn

class ModelEWMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: torch.device = torch.device("cpu")):
        self.decay = decay
        # shadow 保存 EMA 权重
        self.shadow = {k: v.detach().clone().to(device) for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        用当前模型参数更新 EMA 权重
        """
        d = self.decay
        msd = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(d).add_(msd[k], alpha=(1 - d))

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        """
        将 EMA 权重拷贝到给定模型上
        """
        msd = model.state_dict()
        for k, v in self.shadow.items():
            msd[k].copy_(v)