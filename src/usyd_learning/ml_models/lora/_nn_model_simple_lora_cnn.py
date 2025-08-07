import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModel, NNModelArgs
from ...ml_algorithms.lora import LoRALinear


class NNModel_SimpleLoRACNN(NNModel):
    def __init__(self):
        super().__init__()
        self.fc1 = None
        self.bn2 = None
        self.conv2 = None
        self.bn1 = None
        self.conv1 = None

    # override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = LoRALinear(32 * 5 * 5, 10, rank=5)
        return self         # Note: return self

    # override
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
