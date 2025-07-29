from __future__ import annotations

from .fed_node_event_args import FedNodeEventArgs
from .fed_node import FedNode
from .fed_node_type import EFedNodeType

from ..ml_utils import console

# from train_strategy.client_strategy.fedavg_client import FedAvgClientTrainingStrategy
# from model_adaptor.lora_model_weight_adaptor import LoRAModelWeightAdapter
# from model_extractor.advanced_model_extractor import AdvancedModelExtractor

import copy


class FedNodeClient(FedNode):
    def __init__(self, node_id: str, node_group:str = ""):
        super().__init__(node_id, node_group)

        # Client node type
        self.node_type = EFedNodeType.client
        return

    # override
    def _run_once(self):
        return

    #################################################

    def _create_extractor(self):
        """
        " create extractor
        """
        if not hasattr(self.args, "local_lora_model"):
            return None
        if self.args.local_lora_model is None:
            return None
        return AdvancedModelExtractor(self.args.local_lora_model)

    def extract_args(self):
        return 0
        # TODO: make different arg extractor class

    def run(self, **kwargs):
        return 0

    def local_training(self):
        """Client node executes local operations."""
        return self.strategy.run_local_training()

    def observation(self):
        """Client node executes local observation operations."""
        return self.strategy.run_observation()

    def update_weights(self, new_weight):
        self.args.client_weight = copy.deepcopy(new_weight)

    def update_global_weight(self, new_weight):
        # simulate receive global weight and set it to local model
        self.args.global_weight = copy.deepcopy(new_weight)

    def update_local_wbab(self, new_wbab):
        self.args.local_wbab = copy.deepcopy(new_wbab)

    def update_global_wbab(self, new_wbab):
        self.args.global_wbab = copy.deepcopy(new_wbab)

    def apply_wbab(self, new_wbab):
        LoRAModelWeightAdapter.apply_weights_to_model(
            self.args.local_lora_model, new_wbab
        )