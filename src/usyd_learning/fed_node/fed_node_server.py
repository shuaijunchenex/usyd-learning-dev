from __future__ import annotations

import copy


from ..ml_utils import console
from ..ml_data_process import DataDistribution
from .fed_node import FedNode
from .fed_node_type import EFedNodeType

# from train_strategy.client_strategy.fedavg_client import FedAvgClientTrainingStrategy
# from model_adaptor.lora_model_weight_adaptor import LoRAModelWeightAdapter
# from model_extractor.advanced_model_extractor import AdvancedModelExtractor 


class FedNodeServer(FedNode):
    def __init__(self, node_id: str, node_group: str = ""):
        super().__init__(node_id, node_group)

        # Server node type
        self.__node_type = EFedNodeType.server
        return


    # override
    def _run_once(self):
        console.info(f"{self._node_id}: Run...")
        pass

    ##################################################

    def _create_extractor(self):
        """
        " create extractor
        """
        if not hasattr(self.args, 'local_lora_model'):
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

    def create_node_info(self):
        info = super().create_node_info()
        # Add client-specific information
        info.update({
            'role': 'client',
            'train_data': self.train_data,
            'test_data': self.test_data,
            'loss_func': self.loss_func,
            'optimizer': self.optimizer,
        }) 
        return info

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
        LoRAModelWeightAdapter.apply_weights_to_model(self.args.local_lora_model, new_wbab)