from __future__ import annotations

from ml_data_loader.dataset_loader import DatasetLoader
from ml_data_loader.dataset_loader_args import DatasetLoaderArgs
from ml_data_loader.dataset_type import EDatasetType

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
Dataset loader for mnist
'''


class _DatasetLoader_Minst(DatasetLoader):
    def __init__(self):
        super(_DatasetLoader_Minst, self).__init__(EDatasetType.mnist)

    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        #create default transform
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(torch.flatten)])

        train_set = datasets.MNIST(root=args.root, train=True, transform=args.transform, download=args.is_download)
        test_set = datasets.MNIST(root=args.root, train=False, transform=args.transform, download=args.is_download)

        self._train_data_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
        self._test_data_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
        return
