from __future__ import annotations
from abc import ABC, abstractmethod

import torch.nn as nn

from ml_models.nn_model_args_abc import AbstractNNModelArgs, ENNModelType

"""
NN Model virtual class
"""

class AbstractNNModel(ABC, nn.Module):          #虚基类
    def __init__(self, model_type : ENNModelType):
        super().__init__()

        """
        " Attr: Model Type
        """
        self._model_type : ENNModelType = model_type

        """
        " Attr: Model arga
        """
        self._model_args: AbstractNNModelArgs = None
        return

    @property
    def model_type(self) -> ENNModelType:
        """
        " Property: Model Type类型(readonly)
        """
        return self._model_type

    @property
    def model_args(self) -> AbstractNNModelArgs:
        """
        " Property: Model Args参数(readonly)
        """
        return self._model_args

    @abstractmethod
    def create_model(self, args: AbstractNNModelArgs = None) -> AbstractNNModel:
        """
        " Abstract: create model
        """
        pass