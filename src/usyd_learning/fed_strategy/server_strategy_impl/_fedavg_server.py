from __future__ import annotations
from typing import Dict, List, Any, Optional
from collections import defaultdict

import torch
from fed_strategy.server_strategy import ServerStrategy
from fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from model_trainer.model_evaluator import ModelEvaluator


class FedAvgServerStrategy(ServerStrategy):

    def __init__(self) -> None:
        super().__init__()

    def _create_inner(self, args, server_node) -> None:
        self._args = args
        self._strategy_type = "fedavg"
        self._obj = server_node
        return self

    def aggregation(self) -> dict:
        aggregator = self._obj.node_var.aggregation_method
        aggregated_weights = aggregator.aggregate(self.node_var.client_updates) #TODO: check
        self.node_var.global_model_weights = aggregated_weights
        return aggregated_weights

    def receive(self, client_updates) -> None:
        self.node_var.client_updates = client_updates #{client1: {weight:"", data_vol:""}, client2: {weight:"", data_vol:""}}
        raise NotImplementedError

    def broadcast(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> None:
        evaluator = self._obj.node_var.model_evaluator
        test_data = self._obj.node_var.test_data
        raise NotImplementedError

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError