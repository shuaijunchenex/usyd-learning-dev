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

    @abstractmethod
    def select_clients(self, available_clients) -> list:
        """
        Select a subset of clients for the current round.
        :param available_clients: List of available client nodes.
        :return: List of selected client nodes.
        """
        pass

    @abstractmethod
    def record_evaluation(self)-> None:
        """
        Record evaluation metrics.
        """
        pass

    @abstractmethod
    def receive_client_updates(self, client_updates) -> None:
        """
        Receive updates from clients.
        :param client_updates: List of updates from clients.
        """
        pass

    @abstractmethod
    def prepare(self, logger_header, client_nodes_in) -> None:
        """
        Prepare the strategy before starting the training rounds.
        :param logger_header: Header information for logging.
        :param client_nodes_in: List of client nodes to be used in the strategy.
        """
        pass

    @abstractmethod
    def apply_weight(self): 
        """
        Apply the aggregated weights to the server's model.
        """
        pass
