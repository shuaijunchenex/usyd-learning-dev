from __future__ import annotations

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
Dataset loader for mnist
'''
class DatasetLoader_Minst(DatasetLoader):
    def __init__(self):
        super().__init__()

    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        #create default transform
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(torch.flatten)])

        if args.is_load_train_set:
            train_set = datasets.MNIST(root=args.root, train=True, transform=args.transform, download=args.is_download)
            self.train_data_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
        else:
            args.is_load_train_set = False

        if args.is_load_test_set:
            test_set = datasets.MNIST(root=args.root, train=False, transform=args.transform, download=args.is_download)
            self.test_data_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
        else:
            args.is_load_test_set = False

        return
