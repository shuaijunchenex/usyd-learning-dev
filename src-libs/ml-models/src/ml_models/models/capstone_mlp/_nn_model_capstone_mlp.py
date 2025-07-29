import torch.nn as nn

from nn_model import AbstractNNModel, AbstractNNModelArgs, ENNModelType, NNModel

class _NNModel_CapstoneMLP(NNModel):

    """
    " Private class for CapstoneMLP model implementation
    """

    def __init__(self):

        #call super construct, init Model Type
        super(_NNModel_CapstoneMLP, self).__init__(ENNModelType.capstoneMLP)
        

    def create_model(self, args: AbstractNNModelArgs = None) -> AbstractNNModel:
        super().create_model(args)
        
        """
        " Virtual method implementation: Create Capstone MLP
        """
        self._flatten = nn.Flatten()
        self._fc1 = nn.Linear(784, 128)
        self._relu1 = nn.ReLU()
        self._fc2 = nn.Linear(128, 64)
        self._relu2 = nn.ReLU()
        self._fc3 = nn.Linear(64, 10)
        return self         #Note: return self

    def forward(self, x):
        x = self._flatten(x)
        x = self._relu1(self._fc1(x))
        x = self._relu2(self._fc2(x))
        x = self._fc3(x)
        return x
