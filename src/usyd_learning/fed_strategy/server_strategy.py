from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..fed_strategy.strategy_args import StrategyArgs
from ..ml_utils import TrainingLogger, EventHandler, console, String, ObjectMap, KeyValueArgs
from .base_strategy import BaseStrategy

class ServerStrategy(BaseStrategy):

    def __init__(self) -> None:
        super().__init__()
        self._strategy_type: str = "server" 
        self._obj = None

    def create(self, args: StrategyArgs, server_node):
        self._args = args
        self._create_inner(args, server_node)  # create dataset loader

        return self

    @abstractmethod
    def aggregation(self) -> dict:
        """
        Aggregate weights from clients.
        :param client_weights: List of weights from clients.
        :return: Aggregated weights.
        """
        pass

    @abstractmethod
    def broadcast(self) -> None:
        """
        Broadcast aggregated weights to clients.
        :param aggregated_weights: The aggregated weights to be broadcast.
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """
        Main loop/step for the strategy (e.g., one FL round orchestration).
        """
        pass

    @abstractmethod
    def evaluate(self) -> None:
        """
        Evaluate server-side performance/metrics.
        """
        pass
