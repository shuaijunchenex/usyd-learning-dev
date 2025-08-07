from __future__ import annotations
from abc import ABC, abstractmethod

import torch.nn as nn

from .nn_model_args import NNModelArgs

"""
NN Model virtual class
"""

class AbstractNNModel(ABC, nn.Module):      
    def __init__(self):
        super().__init__()

        #Attr: Model args
        self._model_args: NNModelArgs
        return

    @property
    def model_type(self) -> str:
        """
        " Property: Model Type(readonly)
        """
        return self._model_args.model_type

    @property
    def model_args(self) -> NNModelArgs:
        """
        " Property: Model Args(readonly)
        """
        return self._model_args

    @abstractmethod
    def create_args(self) -> NNModelArgs:
        """
        " Create model args
        """
        pass

    @abstractmethod
    def create_model(self, args: NNModelArgs) -> AbstractNNModel:
        """
        " Abstract: create model
        """
        if args is None:
            args = NNModelArgs()

        self._model_args = args
        return self

    @abstractmethod
    def forward(self, x):
        pass
