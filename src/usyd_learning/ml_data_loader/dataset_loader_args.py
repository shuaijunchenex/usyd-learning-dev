from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from ..ml_utils import KeyValueArgs, dict_exists, dict_get
from .dataset_loader_util import DatasetLoaderUtil


@dataclass
class DatasetLoaderArgs(KeyValueArgs):
    """
    Dataset loader arguments
    """

    dataset_type: str = ""  # Dataset type
    root: str = ""          # data set files folder
    split: str = ""
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 4

    # is download from internet
    is_download: bool = True

    # is load training set
    is_load_train_set: bool = True
    is_load_test_set: bool = False

    # Collate and tramsform
    collate_fn: Any = None
    transform: Any = None
    text_collate_fn = DatasetLoaderUtil.text_collate_fn

    def __init__(self, config_dict: dict, is_clone_dict = False):
        super().__init__(config_dict, is_clone_dict)

        if config_dict is None:
            self.set_args({}) 
        elif dict_exists(config_dict, "data_loader|dataset_loader"):
             self.set_args(dict_get(config_dict, "data_loader|dataset_loader"))

        self.dataset_type = self.get("name", "mnist")
        self.root = self.get("root", ".dataset")
        self.split = self.get("split", "")
        self.batch_size = self.get("batch_size", 64)
        self.shuffle = self.get("shuffle", True)
        self.num_workers = self.get("num_workers", 4)
        self.is_download = self.get("is_download", True)

        self.is_load_train_set = self.get("is_load_train_set", True)
        self.is_load_test_set = self.get("is_load_test_set", False)

        return
