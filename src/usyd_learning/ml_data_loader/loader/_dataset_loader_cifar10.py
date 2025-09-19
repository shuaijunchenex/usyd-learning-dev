from __future__ import annotations

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
Dataset loader for Cifar10
'''
class DatasetLoader_Cifar10(DatasetLoader):
    def __init__(self):
        super().__init__()

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        if args.transform is None:
            args.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
        ])

        self._dataset = datasets.CIFAR10(
            root=args.root, train=True,
            transform=args.transform, download=args.is_download
        )
        self._data_loader = DataLoader(
            self._dataset, batch_size=args.batch_size,
            shuffle=args.shuffle, num_workers=args.num_workers
        )

        test_transform = getattr(args, "test_transform", None) or args.transform
        test_batch_size = getattr(args, "test_batch_size", None) or args.batch_size

        self._test_dataset = datasets.CIFAR10(
            root=args.root, train=False,
            transform=test_transform, download=args.is_download
        )
        self._test_data_loader = DataLoader(
            self._test_dataset, batch_size=test_batch_size,
            shuffle=False, num_workers=args.num_workers
        )
        return

