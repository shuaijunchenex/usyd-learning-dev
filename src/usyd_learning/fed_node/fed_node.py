from __future__ import annotations
from abc import ABC, abstractmethod
from copy import copy
from typing import Optional

from .fed_node_vars import FedNodeVars
from .fed_node_type import EFedNodeType
from ..ml_simu_switcher import SimuNode, SimuSwitcher
from ..ml_utils import EventHandler, console, String

'''
Node class interface declare(virtual class as interface)
Note:
   'data_loader_collate_fn', 'data_loader_transform' member for data loader
'''


class FedNode(ABC, EventHandler):
    def __init__(self, node_id: str, node_group: str = ""):
        EventHandler.__init__(self)

        self._node_id: str = node_id  # Unique Node ID
        self.node_group: str = node_group  # Belong to group
        self.node_type: EFedNodeType = EFedNodeType.unknown  # Node Type
        self.simu_switcher: Optional[SimuSwitcher] = None  # switcher
        self.simu_node: Optional[SimuNode] = None  # Simu node of switcher

        # Node var associated to node
        self.node_var: Optional[FedNodeVars] = None
        return

    @property
    def node_id(self):
        return self._node_id

    @property
    def node_full_id(self):
        """
        Node full id with group id, like "client_1@group_1"
        """
        return f"{self._node_id}@{self.node_group}"

    def with_node_var(self, var: FedNodeVars):
        """
        DI node var
        """
        self.node_var = var
        return self

    def create_simu_node(self, simu_switcher: SimuSwitcher):
        """
        Create node's simu node for data exchange
        """
        self.simu_switcher = simu_switcher
        self.simu_node = self.simu_switcher.create_node(self._node_id)
        return

    def connect(self, node_id: str):
        """
        Make connection of this node to specified simu node(node id)
        """
        self.simu_node.connect(node_id)
        return

    @abstractmethod
    def run(self) -> None:
        """
        run node
        """
        return

    def training(self):
        """
        Client node executes local operations.
        """
        return self.strategy.run_local_training()

    def __str__(self):
        return self._node_id

    ##################################################################

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
        LoRAModelWeightAdapter.apply_weights_to_model(self.args.local_lora_model, new_wbab)

    def custimize_local_model(self, custimized_local_model):
        self.custimized_local_model = custimized_local_model

    def set_weights(self, weights):
        """Set model weights using the strategy provided."""
        self.weight_handler.apply_weights(self.model, weights)

    def get_weights(self):
        """Retrieve the current model weights."""
        return self.weight_handler.get_weights(self.model)
