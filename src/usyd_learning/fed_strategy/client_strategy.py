from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
from ..ml_algorithms import OptimizerBuilder
from .strategy_args import StrategyArgs
from .base_strategy import BaseStrategy
#from usyd_learning.fed_node.fed_node import FedNode
try:
    from ..ml_utils import console
except Exception:
    console = None

class ClientStrategy(BaseStrategy):
    """Abstract base for a client's local-training/observation strategy."""

    def __init__(self) -> None:
        super().__init__()
        self._strategy_type : str = "client"
        self._obj = None

    def create(self, args: StrategyArgs, client_node):
        self._args = args
        self._create_inner(args, client_node)  # create dataset loader

        return self

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