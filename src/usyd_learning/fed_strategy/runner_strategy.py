from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Dict, Any, List, Optional
from ..fed_runner.fed_runner import FedRunner
from ..fed_strategy.strategy_args import StrategyArgs
from ..fed_strategy.strategy import BaseStrategy

class RunnerStrategy(BaseStrategy):
    
    def __init__(self, runner: FedRunner, client_node, server_node) -> None:
        self._strategy_type = "runner"
        self._obj = runner
        self._server_node = server_node
        self._client_nodes = client_node
        return

    @abstractmethod
    def run(self, runner: FedRunner) -> None:
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def simulate_client_local_training_process():
        """
        Local training simulation method.
        This method should be overridden by subclasses to implement local training logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def simulate_server_broadcast_process():
        """
        Server broadcast simulation method.
        This method should be overridden by subclasses to implement server broadcast logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def simulate_server_update_process():
        """
        Server update simulation method.
        This method should be overridden by subclasses to implement server update logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
