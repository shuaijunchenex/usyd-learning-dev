from __future__ import annotations

from ml_models.nn_model_abc import AbstractNNModelArgs, AbstractNNModel, ENNModelType

"""
NN Model virtual class
"""

class NNModel(AbstractNNModel):
     def __init__(self, model_type : ENNModelType):
         super().__init__(model_type)


     def create_model(self, args: AbstractNNModelArgs = None) -> AbstractNNModel:
         """
         " Virtual method implementation
         " Subclass implement this function to create NN model
         """
         return super().create_model(args)





