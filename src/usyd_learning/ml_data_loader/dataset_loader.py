from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional

from torch.utils.data import DataLoader, Dataset
from .dataset_loader_args import DatasetLoaderArgs


class DatasetLoader(ABC):
    """
    " Data set loader abstract class
    """

    def __init__(self):
        self._dataset_type: str = ""                # Dataset type
        self._data_loader: Optional[DataLoader] = None    # Training data loader
        self._dataset: Dataset | None = None
        self._args: DatasetLoaderArgs | None = None
        self._after_create_fn: Callable[[DatasetLoader], None] | None = None
        return

    # --------------------------------------------------
    @property
    def dataset_type(self): return self._dataset_type

    @property
    def data_loader(self) -> DataLoader:
        if self._data_loader is not None:
            return self._data_loader 
        else:
            raise ValueError("ERROR: DatasetLoader's data_loader is None.")

    @property
    def data_set(self): return self._dataset

    @property
    def is_loaded(self): return self._data_loader is not None

    @property
    def args(self): return self._args

    # --------------------------------------------------
    def create(self, args: DatasetLoaderArgs, fn: Callable[[DatasetLoader], None]|None = None) -> DatasetLoader:
        """
        Create Dataset Loader
        """
        self._args = args
        self._create_inner(args)  # create dataset loader
        if fn is not None:
            self._after_create_fn = fn
            fn(self)
        return self

    @abstractmethod
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        """
        Real create loader
        """
        pass
