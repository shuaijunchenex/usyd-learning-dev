from abc import ABC, abstractmethod
from typing import Any
import torch.nn as nn

class BaseWeightAdapter(ABC):
    def __init__(self):
        self.source_weights = None

    def set_source_weights(self, weights_dict: dict) -> 'BaseWeightAdapter':
        self.source_weights = weights_dict
        return self

    @abstractmethod
    def apply_to_model(self, model: nn.Module) -> nn.Module:
        pass

    @abstractmethod
    def load_from_file(self, filepath: str) -> 'BaseWeightAdapter':
        pass

    @abstractmethod
    def load_from_extractor(self, extractor: Any) -> 'BaseWeightAdapter':
        pass
