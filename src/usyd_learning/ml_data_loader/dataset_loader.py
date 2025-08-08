from __future__ import annotations
from abc import ABC, abstractclassmethod
from typing import Callable

from torch.utils.data import DataLoader, Dataset
from .dataset_loader_args import DatasetLoaderArgs


class DatasetLoader(ABC):
    """
    " Data set loader abstract class
    """

    def __init__(self):
        self._dataset_type: str  # Dataset type
        self._data_loader: DataLoader|None = None  # Training data loader
        self._dataset: Dataset|None = None
        self._args: DatasetLoaderArgs
        self._after_create_fn: Callable[[DatasetLoader]]|None = None
        return

    # --------------------------------------------------
    @property
    def dataset_type(self): return self._dataset_type

    @property
    def data_loader(self): return self._data_loader

    @property
    def data_set(self): return self._dataset

    @property
    def is_loaded(self): return self._data_loader is not None

    @property
    def args(self): return self._args

    # --------------------------------------------------
    def create(self, args: DatasetLoaderArgs, fn:Callable[[DatasetLoader]] = None) -> DatasetLoader:
        """
        Creta Dataset Loader
        """
        self._args = args
        self._create_inner(args)  # creatte dataset loader
        if fn is not None:
            self._after_create_fn = fn
            fn(self)
        return self

    @abstractclassmethod
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        """
        Real create loader
        """
        pass
