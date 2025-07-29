from __future__ import annotations
from abc import ABC

from ml_models.nn_model_type import ENNModelType

"""
NN Model args interface(ABC class)
"""

class AbstractNNModelArgs(ABC):
    def __init__(self):

        self.model_type: ENNModelType = ENNModelType.unknown
