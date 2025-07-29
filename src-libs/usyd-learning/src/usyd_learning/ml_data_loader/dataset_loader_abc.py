from __future__ import annotations
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from .dataset_loader_args import DatasetLoaderArgs


class AbstractDatasetLoader(ABC):
    """
    " Data set loader abstract class
    """

    def __init__(self):
        self._dataset_type: str  # Dataset type
        self.train_data_loader: DataLoader = None  # Training data loader
        self.test_data_loader: DataLoader = None  # Test data loader
        return

    # --------------------------------------------------
    @property
    def dataset_type(self):
        return self._dataset_type

    @property
    def has_train_loader(self) -> bool:
        return self.train_data_loader is not None

    @property
    def has_test_loader(self) -> bool:
        return self.test_data_loader is not None

    # --------------------------------------------------
    def create(self, args: DatasetLoaderArgs) -> AbstractDatasetLoader:
        """
        Creta Dataset Loader
        """
        self._pre_create(args)  # before create
        self._create_inner(args)  # creatte dataset loader
        self._after_create(args)  # after create
        return self

    @abstractmethod
    def _pre_create(self, args: DatasetLoaderArgs) -> None:
        """
        Call before create loader
        """
        pass

    @abstractmethod
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        """
        Real create loader
        """
        pass

    @abstractmethod
    def _after_create(self, args: DatasetLoaderArgs) -> None:
        """
        Call after loader created
        """
        pass
