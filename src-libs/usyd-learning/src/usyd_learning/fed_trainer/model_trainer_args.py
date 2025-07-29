from __future__ import annotations
from dataclasses import dataclass

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from ..ml_utils.key_value_args import KeyValueArgs

@dataclass
class ModelTrainerArgs(KeyValueArgs):
    """
    Trainer type
    """
    trainer_type: str = "standard"
    
    """
    Torch NN Model
    """
    model: nn.Module = None

    """
    Optimizer
    """
    optimizer: optim.Optimizer = None

    """
    Loss function
    """
    loss_func = None

    """
    Training data
    """
    train_loader: DataLoader = None
    test_loader: DataLoader = None
    
    """
    Run on device via 'cpu' or 'gpu'
    """
    device: str = "cpu"

    """
    """
    save_path: str = ""

    """
    """
    best_val_acc = 0.0

    def __init__(self, config_dict: dict = None, is_clone_dict = False):
        super().__init__(config_dict, is_clone_dict)

        if config_dict is None:
            self.set_args({}) 
        elif "trainer" in config_dict:
             self.dataset_type = self.get("trainer_type", "")
             self.device = self.get("device", "cpu")
             self.device = self.get("save_path", "./.trainer_result/")
