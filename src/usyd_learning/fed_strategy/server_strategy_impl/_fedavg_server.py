from __future__ import annotations
from typing import Dict, List, Any, Optional
from collections import defaultdict

import torch
from fed_strategy.server_strategy import ServerStrategy
from fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from model_trainer.model_evaluator import ModelEvaluator


class FedAvgServerStrategy(ServerStrategy):

    def __init__(self, server_node) -> None:
        super().__init__(server_node)
        self._strategy_type = "FedAvg"

    def _create_inner(self) -> None:
        
        self._server_node.node_var.client_selection
        self._server_node.node_var.aggregation_method

    def aggregation(self) -> dict:
        raise NotImplementedError

    def broadcast(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> None:
        raise NotImplementedError

    def create_server_strategy(self, server_strategy_args, fn=None) -> FedAvgServerStrategy:
        """
        Backward-compatible factory-style entry.
        """
        return self.create(server_strategy_args, fn)

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError