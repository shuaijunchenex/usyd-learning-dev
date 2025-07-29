from __future__ import annotations

from ml_models.nn_model_abc import AbstractNNModelArgs, AbstractNNModel
from ml_models.nn_model_type import ENNModelType

"""
NN Model factory static class
"""

class NNModelFactory:

     @staticmethod
     def create(args: AbstractNNModelArgs) -> AbstractNNModel:
         match args.model_type:
             case ENNModelType.mnist2NNBrenden:
                from ml_models.models.minist_nn_brenden._nn_model_mnist2NNBrenden import _NNModel_Mnist2NNBrenden
                return _NNModel_Mnist2NNBrenden().create_model(args)
             case ENNModelType.capstoneMLP:
                from ml_models.models.capstone_mlp._nn_model_capstone_mlp import _NNModel_CapstoneMLP
                return _NNModel_CapstoneMLP().create_model(args)
             case ENNModelType.simpleMLP:
                from ml_models.models.simple_mlp._nn_model_simple_mlp import _NNModel_SimpleMLP
                return _NNModel_SimpleMLP().create_model(args)

         return None
