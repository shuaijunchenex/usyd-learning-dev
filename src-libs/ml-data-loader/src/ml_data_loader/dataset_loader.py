from __future__ import annotations

from ml_data_loader.dataset_loader_abc import AbstractDatasetLoader
from ml_data_loader.dataset_loader_args import DatasetLoaderArgs
from ml_data_loader.dataset_type import EDatasetType


class DatasetLoader(AbstractDatasetLoader):
    """
    " Data set loader abstract class
    """

    def __init__(self, dataSetType: EDatasetType):
        super().__init__(dataSetType)
        return

    #override
    def _pre_create(self, args: DatasetLoaderArgs) -> None:
        return
    
    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        return

    #override
    def _after_create(self, args: DatasetLoaderArgs) -> None:
        return
