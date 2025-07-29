from __future__ import annotations
from dataclasses import dataclass

from ml_data_loader.dataset_type import EDatasetType


@dataclass
class DatasetLoaderArgs:
    '''
    Dataset loader arguments
    '''

    dataset_type: EDatasetType = EDatasetType.unknown       # Dataset type
    root: str = ""      #data set folder
    split: str = ""
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 4
    collate_fn = None
    transform = None
    is_download: bool = False       #is download from internet


    def __init__(self):
        return
