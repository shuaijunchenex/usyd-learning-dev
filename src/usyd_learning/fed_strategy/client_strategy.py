from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

from ..model_trainer import ModelTrainer, ModelTrainerFactory, ModelTrainerArgs
from ..ml_algorithms import OptimizerBuilder
from .base_strategy import BaseStrategy
try:
    from ..ml_utils import console
except Exception:
    console = None

class ClientStrategy(BaseStrategy):
    """Abstract base for a client's local-training/observation strategy."""

    def __init__(self, client_node) -> None:
        super().__init__()
        self._strategy_type = "client"
        self._obj = client_node

    @abstractmethod
    def run_observation(self): 
        pass

    @abstractmethod
    def run_local_training(self):
        pass

    @abstractmethod
    def observation_step(self):
        pass
        
    @abstractmethod
    def local_training_step(self):
        pass