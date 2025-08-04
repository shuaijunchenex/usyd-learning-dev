from __future__ import annotations

from .dataset_loader_abc import AbstractDatasetLoader
from .dataset_loader_args import DatasetLoaderArgs


class DatasetLoader(AbstractDatasetLoader):
    """
    " Data set loader impl class
    """

    def __init__(self):
        super().__init__()
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
