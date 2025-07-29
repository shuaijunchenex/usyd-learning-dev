from __future__ import annotations
from abc import ABC, abstractmethod

import torch.nn as nn

from ml_simu_switcher.simu_node import SimuNode

from ml_fed_learn.fed_node.ml_fed_node_type import EFedNodeType


'''
' Node class interface declare(virtual class as interface)
'''


class AbstractFedNode(ABC):
    def __init__(self, node_id : int, node_config: dict):
        self.node_id : int = node_id                                # Node ID
        self.node_type : EFedNodeType = EFedNodeType.unknown        # Node Type
        
        #Common var for all Node
        self.node_config = node_config  # general yaml configuration

        self.train_data: any            # Training data (e.g., PyTorch DataLoader)
        self.test_data: any             # Test data (e.g., PyTorch DataLoader)
        self.global_model: nn.Module    # The global model shared across clients
        self.device: str = "cpu"        # Computation device (default: CPU)
        
        self.logger = None              # logger
        self.simu_node : SimuNode = None

    @abstractmethod
    def run(self):
        """
        run node
        """
        pass

    @abstractmethod
    def _parse_config():
        """
        Parse yaml config
        """
        pass

    @abstractmethod
    def _prepare():
        """
        Prepare before run
        """
        pass

    ##################################################################
    def custimize_local_model(self, custimized_local_model):
        self.custimized_local_model = custimized_local_model

    def set_weights(self, weights):
        """Set model weights using the strategy provided."""
        self.weight_handler.apply_weights(self.model, weights)

    def get_weights(self):
        """Retrieve the current model weights."""
        return self.weight_handler.get_weights(self.model)

    def create_node_info(self) -> dict[str, any]:
        """Return common node information, which can be used for aggregation and other operations."""
        return {
            'node_id': self.node_id,
            'epoch_size': self.epoch_size,
            'batch_size': self.batch_size,
            'lr': self.lr,
        }
