from __future__ import annotations

from ml_data_loader.dataset_loader import DatasetLoader
from ml_data_loader.dataset_loader_args import DatasetLoaderArgs
from ml_data_loader.dataset_type import EDatasetType

class DatasetLoaderFactory:
    """
    " Dataset loader factory
    """

    @staticmethod
    def create(dataLoadArgs: DatasetLoaderArgs) -> DatasetLoader:
        """
        " Static method to create data loader
        """
        match dataLoadArgs.dataset_type:
            case EDatasetType.mnist:
                from ml_data_loader.loader._dataset_loader_minst import _DatasetLoader_Minst
                return _DatasetLoader_Minst().create(dataLoadArgs)
            case EDatasetType.fminst:
                from ml_data_loader.loader._dataset_loader_fminst import _DatasetLoader_Fminst
                return _DatasetLoader_Fminst().create(dataLoadArgs)
            case EDatasetType.cifar10:
                from ml_data_loader.loader._dataset_loader_cifar10 import _DatasetLoader_Cifar10
                return _DatasetLoader_Cifar10().create(dataLoadArgs)
            case EDatasetType.cifar100:
                from ml_data_loader.loader._dataset_loader_cifar100 import _DatasetLoader_Cifar100
                return _DatasetLoader_Cifar100().create(dataLoadArgs)
            case EDatasetType.emnist:
                from ml_data_loader.loader._dataset_loader_emnist import _DatasetLoader_Emnist
                return _DatasetLoader_Emnist().create(dataLoadArgs)
            case EDatasetType.kmnist:
                from ml_data_loader.loader._dataset_loader_kmnist import _DatasetLoader_Kmnist
                return _DatasetLoader_Kmnist().create(dataLoadArgs)
            case EDatasetType.qmnist:
                from ml_data_loader.loader._dataset_loader_qmnist import _DatasetLoader_Qmnist
                return _DatasetLoader_Qmnist().create(dataLoadArgs)
            case EDatasetType.stl10:
                from ml_data_loader.loader._dataset_loader_stl10 import _DatasetLoader_Stl10
                return _DatasetLoader_Stl10().create(dataLoadArgs)
            case EDatasetType.svhn:
                from ml_data_loader.loader._dataset_loader_svhn import _DatasetLoader_Svhn
                return _DatasetLoader_Svhn().create(dataLoadArgs)
            case EDatasetType.imagenet:
                from ml_data_loader.loader._dataset_loader_imagenet import _DatasetLoader_ImageNet
                return _DatasetLoader_ImageNet().create(dataLoadArgs)
            case EDatasetType.agnews:
                from ml_data_loader.loader._dataset_loader_agnews import _DatasetLoader_Agnews
                return _DatasetLoader_Agnews().create(dataLoadArgs)
            case EDatasetType.imdb:
                from ml_data_loader.loader._dataset_loader_imdb import _DatasetLoader_Imdb
                return _DatasetLoader_Imdb().create(dataLoadArgs)
            case EDatasetType.dbpedia:
                from ml_data_loader.loader._dataset_loader_dbpedia import _DatasetLoader_DBpedia
                return _DatasetLoader_DBpedia().create(dataLoadArgs)
            case EDatasetType.yahooanswers:
                from ml_data_loader.loader._dataset_loader_yahooanswers import _DatasetLoader_YahooAnswers
                return _DatasetLoader_YahooAnswers().create(dataLoadArgs)
            
        return None
