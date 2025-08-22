from __future__ import annotations

from .fed_node import FedNode
from .fed_node_type import EFedNodeType
from ..fed_strategy.strategy_factory import StrategyFactory
from ..ml_utils import console

# from train_strategy.client_strategy.fedavg_client import FedAvgClientTrainingStrategy
# from model_adaptor.lora_model_weight_adaptor import LoRAModelWeightAdapter
# from model_extractor.advanced_model_extractor import AdvancedModelExtractor


class FedNodeClient(FedNode):
    def __init__(self, node_id: str, node_group:str = ""):
        super().__init__(node_id, node_group)

        # Client node type
        self.node_type = EFedNodeType.client
        return

    # override
    def run(self) -> None:
        return

    def run_local_training(self):
        """
        Run local training on the client node
        """
        console.info(f"{self._node_id}: Running local training...")
        self.client_strategy.run_local_training()
        pass

    def receive_weight(self, new_weight):
        """
        Receive new weight from server
        """
        self.node_var.model_weight = new_weight
        console.info(f"{self._node_id}: Received new weight from server.")
        return
