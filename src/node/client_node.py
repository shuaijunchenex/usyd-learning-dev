from node.inode import AbstractNode
from train_strategy.client_strategy.fedavg_client import FedAvgClientTrainingStrategy
from model_adaptor.lora_model_weight_adaptor import LoRAModelWeightAdapter
from model_extractor.advanced_model_extractor import AdvancedModelExtractor 

import copy

class ClientNode(AbstractNode):
    def __init__(self, node_id, client_args):
        # Only call the common initialization logic
        super().__init__(node_id)

        # init args
        self.args = client_args
        self.strategy = client_args.strategy(self)
        self.extractor = self._create_extractor()

    def _create_extractor(self):
        """create extractor"""
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