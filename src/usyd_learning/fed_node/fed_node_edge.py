from __future__ import annotations

from .fed_node import FedNode
from .fed_node_type import EFedNodeType

# from train_strategy.client_strategy.fedavg_client import FedAvgClientTrainingStrategy
# from model_adaptor.lora_model_weight_adaptor import LoRAModelWeightAdapter
# from model_extractor.advanced_model_extractor import AdvancedModelExtractor

import copy


class FedNodeEdge(FedNode):
    def __init__(self, node_id: str, node_group:str = ""):
        super().__init__(node_id, node_group)

        # Edge node type
        self.__node_type = EFedNodeType.edge
        
        # Declare edge variables here
        #----------------------------------------
        self.client_node = None
        self.server_node = None
        #----------------------------------------


