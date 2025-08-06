from typing import Any
import torch.nn as nn
import torch.nn.functional as F

from .. import AbstractNNModel, NNModelArgs, NNModel

class NNModel_MnistNNBrenden(NNModel):

    """
    " Private class for Mnist2NNBrenden model implementation
    """

    def __init__(self):
        super().__init__()

    #override
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        super().create_model(args)
        
        self._relu = nn.ReLU()
        self._fc1 = nn.Linear(784, 200)
        self._fc2 = nn.Linear(200, 200)
        self._fc3 = nn.Linear(200, 10)
        return self         #Note: return self

    #override
    def forward(self, x)-> Any:
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = F.softmax(self._fc3(x), dim=1)
        return x
