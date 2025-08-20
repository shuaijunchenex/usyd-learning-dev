from __future__ import annotations

from ..ml_utils import console
from ..ml_data_process import DataDistribution
from .fed_node import FedNode
from .fed_node_type import EFedNodeType
from ..fed_strategy.strategy_factory import StrategyFactory

# from train_strategy.client_strategy.fedavg_client import FedAvgClientTrainingStrategy
# from model_adaptor.lora_model_weight_adaptor import LoRAModelWeightAdapter
# from model_extractor.advanced_model_extractor import AdvancedModelExtractor 


class FedNodeServer(FedNode):
    def __init__(self, yaml, node_id: str, node_group: str = ""):
        super().__init__(node_id, node_group)

        # Server node type
        self.node_type = EFedNodeType.server
        self.strategy_args = StrategyFactory.create_args(yaml)
        self.server_strategy = StrategyFactory.create_server_strategy(self.strategy_args)

        return

    # override
    def run(self) -> None:
        console.info(f"{self._node_id}: Run...")
        pass

    def boroadcast_weight(self):
        return
    
    def update_weight(self, new_weight):
        self.node_var.model_weight = new_weight

