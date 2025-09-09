from __future__ import annotations
from typing import Dict, List, Any, Optional
from collections import defaultdict

import torch
from usyd_learning.fed_strategy.server_strategy import ServerStrategy
from usyd_learning.fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from usyd_learning.fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from usyd_learning.model_trainer.model_evaluator import ModelEvaluator
from usyd_learning.ml_utils import console
from usyd_learning.ml_algorithms.lora.lora_utils import LoRAUtils

class RblaServerStrategy(ServerStrategy):

    def __init__(self, args, server_node) -> None:
        super().__init__()
        self._args = args
        self._strategy_type = "fedavg"
        self._obj = server_node

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

    def broadcast(self, broadcast_objects) -> None:
        for client in broadcast_objects:
            client.set_local_weight
            #client.node_var.model_weight = self._obj.node_var.model_weight
        return

    def sp_svd_split(self, aggregated_weight) -> dict:
        new_weight = LoRAUtils.rbla_split_global_weight(aggregated_weight, LoRAUtils.get_lora_ranks(self._obj.node_var.model))
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def evaluate(self, round) -> None:
        evaluation_dict =  self._obj.node_var.model_evaluator.evaluate()
        evaluation_dict = {"round": round, **evaluation_dict}
        self._obj.node_var.model_evaluator.print_results()
        console.info("Server Evaluation Completed.\n")

        return evaluation_dict

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError