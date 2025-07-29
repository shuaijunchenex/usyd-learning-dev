from __future__ import annotations
from abc import ABC, abstractmethod

from ml_data_loader.dataset_loader_args import DatasetLoaderArgs
from ml_data_loader.dataset_type import EDatasetType

from torch.utils.data import DataLoader


class AbstractDatasetLoader(ABC):
    """
    " Data set loader abstract class
    """

    def __init__(self, dataSetType: EDatasetType):

        #Dataset type
        self._dataset_type: EDatasetType = dataSetType   
        
        #Training DataLoader
        self._train_data_loader: DataLoader = None       

        #Test DataLoader
        self._test_data_loader: DataLoader = None        
        return


    #############################################################
    @property
    def dataset_type(self) -> EDatasetType:
        """
        Data set type
        """
        return self._dataset_type


    #############################################################
    @property
    def train_data_loader(self) -> DataLoader:
        """
        Train DataLoad property(Readonly)
        """
        return self._train_data_loader

    @property
    def has_train_data_loader(self) -> bool:
        """
        Has train DataLoader property(Readonly)
        """
        return self._train_data_loader is not None

    #############################################################
    @property
    def test_data_loader(self) -> DataLoader:
        '''
        Test DataLoad property
        '''
        return self._test_data_loader

    @property
    def has_test_data_loader(self) -> bool:
        '''
        Has test DataLoader
        '''
        return self._test_data_loader is not None

    #############################################################
    def create(self, args: DatasetLoaderArgs) -> AbstractDatasetLoader:
        """
        Creta Dataset Loader
        """
        self._pre_create(args)        # before create
        self._create_inner(args)      # creatte dataset loader
        self._after_create(args)      # after create
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
