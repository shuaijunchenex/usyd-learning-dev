from __future__ import annotations


################
# Node data
################


class SimuNodeData:
    def __init__(self, data: any, from_id: str, to_id: str):
        """
        " from node id
        """
        self.from_node_id: str = from_id

        """
        " to node id
        """
        self.to_node_id: str = to_id

        """
        " data
        """
        self.data: any = data

        """
        " data type
        """
        self._data_type = type(self.data)


    @property
    def data_type(self):
        return self._data_type


    def __str__(self):
        return "NodeData: " + self.from_node_id + "->" + self.to_node_id