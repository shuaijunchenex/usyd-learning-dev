from __future__ import annotations

from .nn_model_abc import NNModelArgs, AbstractNNModel

"""
NN Model virtual class
"""

class NNModel(AbstractNNModel):
     def __init__(self):
         super().__init__()

     #override
     def create_model(self, args: NNModelArgs = None) -> AbstractNNModel:
         return super().create_model(args)

     #override
     def forward(self, x):
        return
