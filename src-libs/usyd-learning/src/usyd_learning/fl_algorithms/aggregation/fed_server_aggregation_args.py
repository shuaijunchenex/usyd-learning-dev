from __future__ import annotations
from dataclasses import dataclass

from ...ml_utils.key_value_args import KeyValueArgs

'''
Dataset loader arguments
'''

@dataclass
class AggregatorArgs(KeyValueArgs):
    """
    " Dataset loader arguments
    """
    type: str = ""
    is_wbab: bool = False # for RBLA use

    def __init__(self, config_dict: dict[str, any], is_clone_dict = False):
        """
        Args for aggregation methods
        """

        super().__init__(config_dict, is_clone_dict)
        if config_dict is None:
            self._key_value_dict = {}
        elif "aggregation" in config_dict:
            self._key_value_dict = config_dict["aggregation"]

        self.type: str = ""

        self.is_wbab: bool = False # for RBLA use
