import torch.nn as nn

from models.simple_lora.nn_model_simple_lora_mlp_args import NNModelArgs_SimpleLoRAMLP
from nn_model import AbstractNNModel, AbstractNNModelArgs, ENNModelType, NNModel

from ..lora.impl.lora_linear import LoRALinear

class _NNModel_SimpleLoRAMLP(NNModel):

    """
    " Private class for SimpleLoRA model implementation
    """

    def __init__(self):

        #call super construct, init Model Type
        super(_NNModel_SimpleLoRAMLP, self).__init__(ENNModelType.simpleLoRAMLP)
        

    def create_model(self, args: AbstractNNModelArgs = None) -> AbstractNNModel:
        super().create_model(args)
        
        """
        " Virtual method implementation
        """

        if(not isinstance(args, NNModelArgs_SimpleLoRAMLP)):
            raise Exception("args is not a NNModelArgs_SimpleLoRA type variable.")

        self._fc1 = LoRALinear(args.input_dim, args.hidden_dim, rank = int(160 * args.rank_ratio), lora_mode=args.lora_mode)
        self._relu = nn.ReLU()
        self._fc2 = LoRALinear(args.hidden_dim, args.hidden_dim, rank = int(100 * args.rank_ratio), lora_mode=args.lora_mode)
        self._relu = nn.ReLU()
        self._fc3 = LoRALinear(args.hidden_dim, args.output_dim, rank = int(10 * args.rank_ratio), lora_mode=args.lora_mode)
        return self         #Note: return self

    def forward(self, x):
        x = self._relu(self._fc1(x))
        x = self._fc2(x)
        x = self._fc3(x)
        return x
