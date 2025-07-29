from __future__ import annotations

import copy

from ml_fed_learn.fed_node.ml_fed_node import FedNode
from ml_fed_learn.fed_node.ml_fed_node_type import EFedNodeType

# from train_strategy.client_strategy.fedavg_client import FedAvgClientTrainingStrategy
# from model_adaptor.lora_model_weight_adaptor import LoRAModelWeightAdapter
# from model_extractor.advanced_model_extractor import AdvancedModelExtractor 


class FedNodeServer(FedNode):
    def __init__(self, node_id, client_config):
        super().__init__(node_id, client_config)

        self.node_type = EFedNodeType.server

        # init class vars
        self.strategy = client_config.strategy(self)
        self.extractor = self._create_extractor()
        self.global_weight = self.global_model.state_dict() if self.global_model else None
        self.evaluator = ModelEvaluator(self.args.global_model, self.args.test_data)
        self.global_model_WbAB = self.extractor.get_layer_dict()

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