from __future__ import annotations
from abc import ABC, abstractmethod
from dataset.dataset_loader_args import DatasetLoaderArgs
from dataset.dataset_type import EDatasetType

from torch.utils.data import DataLoader

'''
Data set loader abstract class
'''


class DatasetLoader(ABC):
    def __init__(self, dataSetType: EDatasetType):

        #数据集类型
        self._dataset_type: EDatasetType = dataSetType   
        
        #训练集DataLoader属性
        self._train_data_loader: DataLoader = None       

        #测试集DataLoader属性
        self._test_data_loader: DataLoader = None        
        return

    #############################################################
    @property
    def dataset_type(self) -> EDatasetType:
        """
        Data set类型
        """
        return self._dataset_type

    #############################################################
    @property
    def train_data_loader(self) -> DataLoader:
        """
        训练集DataLoad property只读
        """
        return self._train_data_loader

    @property
    def has_train_data_loader(self) -> bool:
        """
        是否有训练集DataLoader property只读
        """
        return self._train_data_loader is not None

    #############################################################
    @property
    def test_data_loader(self) -> DataLoader:
        '''
        测试集DataLoad property
        '''
        return self._test_data_loader

    @property
    def has_test_data_loader(self) -> bool:
        '''
        是否有测试集DataLoader
        '''
        return self._test_data_loader is not None

    #############################################################
    def create(self, args: DatasetLoaderArgs) -> DatasetLoader:
        """
        创建Dataset Loader方法
        """
        self._create_inner(args)      #实际创建dataset loader
        self._after_create(args)      #创建后处理
        return self
    
    @abstractmethod
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        pass

    @abstractmethod
    def _after_create(self, args: DatasetLoaderArgs) -> None:
        pass
