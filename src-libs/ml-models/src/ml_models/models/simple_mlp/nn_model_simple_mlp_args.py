from __future__ import annotations
from dataclasses import dataclass

from nn_model_args import NNModelArgs

"""
NN Model args for SimpleMLP
"""

@dataclass
class NNModelArgs_SimpleMLP(NNModelArgs):

    dim_in : int
    dim_hidden : int
    dim_out : int
    softmax_dim : int = 1

    
