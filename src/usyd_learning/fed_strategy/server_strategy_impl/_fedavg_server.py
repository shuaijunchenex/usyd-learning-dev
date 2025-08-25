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
        raise NotImplementedError

    def broadcast(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def evaluate(self) -> None:
        raise NotImplementedError

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError