import torch.nn as nn

from .. import AbstractNNModel, NNModel, NNModelArgs

from ...ml_algorithms.lora import LoRALinear

class NNModel_SimpleLoRAMLP(NNModel):

    """
    " Private class for SimpleLoRA model implementation
    """

    def __init__(self):
        super().__init__()
        

    #override
    def create_model(self, args: NNModelArgs = None) -> AbstractNNModel:
        super().create_model(args)
        
        self._fc1 = LoRALinear(args.input_dim, args.hidden_dim, rank = int(160 * args.rank_ratio), lora_mode=args.lora_mode)
        self._relu = nn.ReLU()
        self._fc2 = LoRALinear(args.hidden_dim, args.hidden_dim, rank = int(100 * args.rank_ratio), lora_mode=args.lora_mode)
        self._relu = nn.ReLU()
        self._fc3 = LoRALinear(args.hidden_dim, args.output_dim, rank = int(10 * args.rank_ratio), lora_mode=args.lora_mode)
        return self         #Note: return self

    #override
    def forward(self, x):
        x = self._relu(self._fc1(x))
        x = self._fc2(x)
        x = self._fc3(x)
        return x
