import torch.nn as nn
import torch.nn.functional as F

from nn_model import AbstractNNModel, AbstractNNModelArgs, ENNModelType, NNModel

class _NNModel_Mnist2NNBrenden(NNModel):

    """
    " Private class for Mnist2NNBrenden model implementation
    """

    def __init__(self):

        #call super construct, init Model Type
        super(_NNModel_Mnist2NNBrenden, self).__init__(ENNModelType.mnist2NNBrenden)


    def create_model(self, args: AbstractNNModelArgs = None) -> AbstractNNModel:
        super().create_model(args)
        
        """
        " Virtual method implementation: Create Minst Brenden NN model
        """
        self._relu = nn.ReLU()
        self._fc1 = nn.Linear(784, 200)
        self._fc2 = nn.Linear(200, 200)
        self._fc3 = nn.Linear(200, 10)
        return self         #Note: return self

    def forward(self, x):
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = F.softmax(self._fc3(x), dim=1)
        return x
