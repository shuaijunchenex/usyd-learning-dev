from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from ..ml_utils.key_value_args import KeyValueArgs

"""
NN Model args
"""

@dataclass
class NNModelArgs(KeyValueArgs):
    # Common
    model_type: str = ""
    share_model: bool = True

    # Specified
    input_dim: int = 0
    hidden_dim: int = 0
    output_dim: int = 0
    softmax_dim : int = 1

    rank_ratio: int = 1
    lora_mode: str = "standard"


    def __init__(self, config_dict: dict[str, Any]|None = None, is_clone_dict = False):
        """
        Model type enum
        """

        if config_dict is None:
            super().__init__({}, is_clone_dict)    
            self.set_args({})
        else:
            super().__init__(config_dict, is_clone_dict)
            if "nn_model" in config_dict:
                self.set_args(config_dict["nn_model"])

        self.model_type = self.get("name", "")
        self.input_dim = self.get("input_dim", 0)
        self.hidden_dim = self.get("hidden_dim", 0)
        self.output_dim = self.get("output_dim", 0)
        self.rank_ratio = self.get("rank_ratio", 0)
        self.lora_mode = self.get("lora_mode", "")
        self.softmax_dim = self.get("softmax_dim", 1)

        self.share_model = self.get("share_model", True)
        return
