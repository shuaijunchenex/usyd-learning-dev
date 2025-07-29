from __future__ import annotations
from dataclasses import dataclass

from ...ml_utils.key_value_args import KeyValueArgs

"""
Data handler args
"""

@dataclass
class DataHandlerArgs(KeyValueArgs):
    type: str = ""

    data_volum_list = None

    verify_allocate: bool = True

    distribution: str = "mnist_lt"

    batch_size: int = 64

    shuffle: bool = False

    num_workers: int = 0

    def __init__(self, config_dict: dict = None, is_clone_dict = False):
        super().__init__(config_dict, is_clone_dict)

        if config_dict is None:
            self.set_args({})
        elif "data_handler" in config_dict:
            self.set_args(config_dict["data_handler"])

        self.type = self.get("type", "")
        self.data_volum_list = self.get("data_volum_list", None)
        self.verify_allocate = self.get("verify_allocate", True)
        self.distribution = self.get("distribution", "mnist_lt")
        self.batch_size = self.get("batch_size", 64)
        self.shuffle = self.get("shuffle", False)
        return    
