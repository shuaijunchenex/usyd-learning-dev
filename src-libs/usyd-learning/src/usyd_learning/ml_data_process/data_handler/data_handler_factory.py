from __future__ import annotations

from .data_handler import DataHandler
from .data_handler_args import DataHandlerArgs

"""
NN Model factory static class
"""

class DataHandlerFactory:
    
     @staticmethod
     def create_args(config_dict: dict = None, is_clone_dict = False) -> DataHandlerArgs:
         return DataHandlerArgs(config_dict, is_clone_dict)

     @staticmethod
     def create(args: DataHandlerArgs) -> DataHandler:
         match args.type:
             case "noniid":
                from .models._nn_model_mnist_nn_brenden import NNModel_MnistNNBrenden
                return NNModel_MnistNNBrenden().create_model(args)
             case "capstone_mlp":
                from .models._nn_model_capstone_mlp import NNModel_CapstoneMLP
                return NNModel_CapstoneMLP().create_model(args)
             case "simple_mlp":
                from .models._nn_model_simple_mlp import NNModel_SimpleMLP
                return NNModel_SimpleMLP().create_model(args)
             case "cifar_convnet":
                from .models._nn_model_cifar_convnet import NNModel_CifarConvnet
                return NNModel_CifarConvnet().create_model(args)
             case "simple_lora_mlp":
                from .models._nn_model_simple_lora_mlp import NNModel_SimpleLoRAMLP
                return NNModel_SimpleLoRAMLP().create_model(args)

         return None
