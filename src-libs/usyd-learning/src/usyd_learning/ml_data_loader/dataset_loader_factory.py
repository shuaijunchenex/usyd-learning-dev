from __future__ import annotations

from .dataset_loader import DatasetLoader
from .dataset_loader_args import DatasetLoaderArgs

class DatasetLoaderFactory:
    """
    " Dataset loader factory
    """

    @staticmethod
    def create_args(config_dict: dict, is_clone_dict: bool = False) -> DatasetLoaderArgs:
        """
        " Static method to create data loader args
        """
        return DatasetLoaderArgs(config_dict, is_clone_dict)

    @staticmethod
    def create(data_loader_args: DatasetLoaderArgs) -> DatasetLoader:
        """
        " Static method to create data loader
        """
        match data_loader_args.dataset_type.lower():
            case "mnist":
                from .loader._dataset_loader_minst import DatasetLoader_Minst
                return DatasetLoader_Minst().create(data_loader_args)
            case "fminst":
                from .loader._dataset_loader_fminst import DatasetLoader_Fminst
                return DatasetLoader_Fminst().create(data_loader_args)
            case "cifar10":
                from .loader._dataset_loader_cifar10 import DatasetLoader_Cifar10
                return DatasetLoader_Cifar10().create(data_loader_args)
            case "cifar100":
                from .loader._dataset_loader_cifar100 import DatasetLoader_Cifar100
                return DatasetLoader_Cifar100().create(data_loader_args)
            case "emnist":
                from .loader._dataset_loader_emnist import DatasetLoader_Emnist
                return DatasetLoader_Emnist().create(data_loader_args)
            case "kmnist":
                from .loader._dataset_loader_kmnist import DatasetLoader_Kmnist
                return DatasetLoader_Kmnist().create(data_loader_args)
            case "qmnist":
                from .loader._dataset_loader_qmnist import DatasetLoader_Qmnist
                return DatasetLoader_Qmnist().create(data_loader_args)
            case "stl10":
                from .loader._dataset_loader_stl10 import DatasetLoader_Stl10
                return DatasetLoader_Stl10().create(data_loader_args)
            case "svhn":
                from .loader._dataset_loader_svhn import DatasetLoader_Svhn
                return DatasetLoader_Svhn().create(data_loader_args)
            case "imagenet":
                from .loader._dataset_loader_imagenet import DatasetLoader_ImageNet
                return DatasetLoader_ImageNet().create(data_loader_args)
            case "agnews":
                from .loader._dataset_loader_agnews import DatasetLoader_Agnews
                return DatasetLoader_Agnews().create(data_loader_args)
            case "imdb":
                from .loader._dataset_loader_imdb import DatasetLoader_Imdb
                return DatasetLoader_Imdb().create(data_loader_args)
            case "dbpedia":
                from .loader._dataset_loader_dbpedia import DatasetLoader_DBpedia
                return DatasetLoader_DBpedia().create(data_loader_args)
            case "yahooanswers":
                from .loader._dataset_loader_yahooanswers import DatasetLoader_YahooAnswers
                return DatasetLoader_YahooAnswers().create(data_loader_args)
            
        return None
