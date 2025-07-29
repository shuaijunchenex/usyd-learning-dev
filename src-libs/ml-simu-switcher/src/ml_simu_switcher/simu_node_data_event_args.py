from __future__ import annotations

from ml_simu_switcher.event_handler.event_args import EventArgs
from ml_simu_switcher.simu_node import SimuNodeData

###
# Simulation node data event args
###

class SimuNodeDataEventArgs(EventArgs, SimuNodeData):
    def __init__(self, data: any, from_id: str, to_id: str):
        EventArgs.__init__(self)
        SimuNodeData.__init__(self, data, from_id, to_id)

    def __str__(self):
        return "NodeEventArgs: " + self.from_node_id + "->" + self.to_node_id