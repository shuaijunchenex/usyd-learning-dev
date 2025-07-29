from __future__ import annotations

import sys
sys.path.insert(0, '')

from src.dataset.dataset_loader import DatasetLoader
from src.dataset.dataset_loader_args import DatasetLoaderArgs
from src.dataset.dataset_type import EDatasetType

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
Dataset loader for mnist
'''


class DatasetLoader_Minst(DatasetLoader):
    def __init__(self):
        super(DatasetLoader_Minst, self).__init__(EDatasetType.mnist)

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        '''
        继承虚方法实现
        '''
        #默认transform
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(torch.flatten)])

        train_set = datasets.MNIST(root=args.root, train=True, transform=args.transform, download=args.is_download)
        test_set = datasets.MNIST(root=args.root, train=False, transform=args.transform, download=args.is_download)

        self._train_data_loader = DataLoader(train_set, args.batch_size, args.shuffle, args.num_workers)
        self._test_data_loader = DataLoader(test_set, args.batch_size, args.shuffle, args.num_workers)
        return

    def _after_create(self, args: DatasetLoaderArgs) -> None:
        '''
        继承虚方法实现
        '''
        return
    
if __name__ == '__main__':
    args = DatasetLoaderArgs()
    args.root = 'C:/MyPhD/Torchly/dataset'
    args.is_download = True
    loader = DatasetLoader_Minst()
    loader.create(args)
    loader.dataset_type()
    print("END")
    
