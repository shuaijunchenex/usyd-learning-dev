from __future__ import annotations
from dataset.dataset_type import EDatasetType

'''
Dataset loader arguments
'''


class DatasetLoaderArgs:
    def __init__(self):
        """
        数据集类型
        """
        self.dataset_type: EDatasetType = EDatasetType.unknown

        #############################################################
        """
        数据集文件路径或url
        """
        self.root: str

        """
        split
        """
        self.split: str

        '''
        Batch size
        '''
        self.batch_size: int = 64

        '''
        shuffle
        '''
        self.shuffle: bool = True

        '''
        number of worker
        '''
        self.num_workers: int = 4

        '''
        call back function
        '''
        self.collate_fn = None

        '''
        Transform, data preprocessing
        '''
        self.transform = None

        '''
        is download from internet
        '''
        self.is_download: bool = False
