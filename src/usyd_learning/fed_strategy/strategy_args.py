from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from ..ml_utils.key_value_args import KeyValueArgs

@dataclass
class StrategyArgs(KeyValueArgs):

    """
    Strategy type
    """
    strategy_obj: str = "client" # or "server"

    """
    strategy mode
    """
    strategy_type: str = "fedavg" # "fedavg", "fedprox", "scaffold", etc.

    def __init__(self, config_dict: dict|None = None, is_clone_dict = False):
        super().__init__(config_dict, is_clone_dict)

        if config_dict is not None and "strategy" in config_dict:
            self.set_args(config_dict["strategy"], is_clone_dict)

        self.strategy_type = self.get("strategy_type", "fedavg")
        self.strategy_obj = self.get("strategy_obj", "client")

        return
