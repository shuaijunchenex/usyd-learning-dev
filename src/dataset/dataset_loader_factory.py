from __future__ import annotations

from dataset.dataset_loader import DatasetLoader
from dataset.dataset_loader_args import DatasetLoaderArgs
from dataset.dataset_type import EDatasetType

from dataset.loader.dataset_loader_agnews import DatasetLoader_Agnews
from dataset.loader.dataset_loader_cifar10 import DatasetLoader_Cifar10
from dataset.loader.dataset_loader_cifar100 import DatasetLoader_Cifar100
from dataset.loader.dataset_loader_dbpedia import DatasetLoader_DBpedia
from dataset.loader.dataset_loader_emnist import DatasetLoader_Emnist
from dataset.loader.dataset_loader_fminst import DatasetLoader_Fminst
from dataset.loader.dataset_loader_imagenet import DatasetLoader_ImageNet
from dataset.loader.dataset_loader_imdb import DatasetLoader_Imdb
from dataset.loader.dataset_loader_kmnist import DatasetLoader_Kmnist
from dataset.loader.dataset_loader_minst import DatasetLoader_Minst
from dataset.loader.dataset_loader_qmnist import DatasetLoader_Qmnist
from dataset.loader.dataset_loader_stl10 import DatasetLoader_Stl10
from dataset.loader.dataset_loader_svhn import DatasetLoader_Svhn
from dataset.loader.dataset_loader_yahooanswers import DatasetLoader_YahooAnswers

'''
Dataset loader factory
'''


class DatasetLoaderFactory:

    @staticmethod
    def create(dataLoadArgs: DatasetLoaderArgs) -> DatasetLoader:
        match dataLoadArgs.dataset_type:
            case EDatasetType.mnist:
                return DatasetLoader_Minst().create(dataLoadArgs)
            case EDatasetType.fminst:
                return DatasetLoader_Fminst().create(dataLoadArgs)
            case EDatasetType.cifar10:
                return DatasetLoader_Cifar10().create(dataLoadArgs)
            case EDatasetType.cifar100:
                return DatasetLoader_Cifar100().create(dataLoadArgs)
            case EDatasetType.emnist:
                return DatasetLoader_Emnist().create(dataLoadArgs)
            case EDatasetType.kmnist:
                return DatasetLoader_Kmnist().create(dataLoadArgs)
            case EDatasetType.qmnist:
                return DatasetLoader_Qmnist().create(dataLoadArgs)
            case EDatasetType.stl10:
                return DatasetLoader_Stl10().create(dataLoadArgs)
            case EDatasetType.svhn:
                return DatasetLoader_Svhn().create(dataLoadArgs)
            case EDatasetType.imagenet:
                return DatasetLoader_ImageNet().create(dataLoadArgs)
            case EDatasetType.agnews:
                return DatasetLoader_Agnews().create(dataLoadArgs)
            case EDatasetType.imdb:
                return DatasetLoader_Imdb().create(dataLoadArgs)
            case EDatasetType.dbpedia:
                return DatasetLoader_DBpedia().create(dataLoadArgs)
            case EDatasetType.yahooanswers:
                return DatasetLoader_YahooAnswers().create(dataLoadArgs)
            
        return None
