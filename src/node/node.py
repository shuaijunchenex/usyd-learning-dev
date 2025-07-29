from abc import ABC, abstractmethod
from node import ENodeType
from __future__ import annotations

'''
Node class interface declare(virtual class as interface)
'''
class INode(ABC):
    def __init__(self, node_id : int):
        self.node_id : int = node_id                           #节点ID
        self.node_type : ENodeType = ENodeType.Unknown         #节点类型
        self._parent_node : INode = None                       #父节点
        self._child_nodes : dict[int, INode] = {}              #子节点Dict <node_id, node>

    @property
    def is_server(self) -> bool: 
        '''
        是否是Server Node
        '''
        return self.node_type == ENodeType.Server

    @property
    def is_edge(self) -> bool: 
        '''
        是否是Edge Node
        '''
        return self.node_type == ENodeType.Edge

    @property
    def is_edge(self) -> bool: 
        '''
        是否是Client Node
        '''
        return self.node_type == ENodeType.Client

    @property    
    def parent_node(self) -> INode:         
        '''
        parent_node属性
        '''
        return self._parent_node

    def with_parent_node(self, node) -> INode:
        '''
        指定节点的父节点 
        '''
        self._parent_node = node
        self.__get_node_type()
        return self

    def add_child_node(self, node) -> INode:
        '''
        增加指定子节点
        '''
        self._child_nodes.update({node.node_id, node})
        self.__get_node_type()
        return self
    
    def __get_node_type(self) -> ENodeType:
        if self._parent_node == None: 
          self.node_type = ENodeType.Server
        elif len(self._child_nodes) == 0:
          self.node_type = ENodeType.Client
        else:
          self.node_type = ENodeType.Edge

    @abstractmethod
    def run(self):
        """
        Abstract method to be implemented by subclasses to define their behavior.
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
