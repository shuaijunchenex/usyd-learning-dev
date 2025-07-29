import torch.nn as nn

from models.simple_mlp.nn_model_simple_mlp_args import NNModelArgs_SimpleMLP
from nn_model import AbstractNNModel, AbstractNNModelArgs, ENNModelType, NNModel

class _NNModel_SimpleMLP(NNModel):

    """
    " Private class for SimpleMLP model implementation
    """

    def __init__(self):

        #call super construct, init Model Type
        super(_NNModel_SimpleMLP, self).__init__(ENNModelType.simpleMLP)
        

    def create_model(self, args: AbstractNNModelArgs = None) -> AbstractNNModel:
        super().create_model(args)
        
        """
        " Virtual method implementation
        """

        if(not isinstance(args, NNModelArgs_SimpleMLP)):
            raise Exception("args is not a NNModelArgs_SimpleMLP type variable.")

        self._layer_input = nn.Linear(args.dim_in, args.dim_hidden)
        self._relu = nn.ReLU()
        self._dropout = nn.Dropout()
        self._layer_hidden = nn.Linear(args.dim_hidden, args.dim_out)
        self._softmax = nn.Softmax(args.softmax_dim)
        return self         #Note: return self

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self._layer_input(x)
        x = self._dropout(x)
        x = self._relu(x)
        x = self._layer_hidden(x)
        return self._softmax(x)
