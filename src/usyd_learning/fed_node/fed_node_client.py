from __future__ import annotations

from .fed_node import FedNode
from .fed_node_type import EFedNodeType

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

