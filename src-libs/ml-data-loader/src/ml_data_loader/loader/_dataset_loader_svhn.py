from __future__ import annotations

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from dataset_loader import DatasetLoader
from dataset_loader_args import DatasetLoaderArgs
from dataset_type import EDatasetType


'''
Dataset loader for svhn
'''


class _DatasetLoader_Svhn(DatasetLoader):
    def __init__(self):
        super(_DatasetLoader_Svhn, self).__init__(EDatasetType.svhn)

    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset = datasets.SVHN(root = args.root, split = args.split, transform = args.transform, download = args.is_download)
        self._train_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
        return
