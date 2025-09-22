from __future__ import annotations

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

'''
Dataset loader for Qmnist
'''
class DatasetLoader_Qmnist(DatasetLoader):
    def __init__(self):
        super().__init__()

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        # 默认 transform（与 MNIST 一致）：ToTensor -> Normalize -> Flatten
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(torch.flatten),
            ])

        self.data_sample_num = 110000
        self.task_type = "cv"

        # torchvision.datasets.QMNIST 使用 "what" 指定 split（常见取值："train"|"test"|"test10k"|"test50k"|"nist"）
        # 若调用方没有显式给出 qmnist_what，则根据 is_train 自动选择 "train"/"test"
        what_train = getattr(args, "qmnist_what", None) or ("train" if args.is_train else "test")

        self._dataset = datasets.QMNIST(
            root=args.root,
            what=what_train,
            transform=args.transform,
            download=args.is_download,
            compat=True,  # 维持与 MNIST 标签的兼容（常见做法）
        )
        self._data_loader = DataLoader(
            self._dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
        )

        # 测试集 loader：允许单独指定 qmnist_what_test，否则默认 "test"
        test_transform = getattr(args, "test_transform", None) or args.transform
        test_batch_size = getattr(args, "test_batch_size", None) or args.batch_size
        what_test = getattr(args, "qmnist_what_test", None) or "test"

        self._test_dataset = datasets.QMNIST(
            root=args.root,
            what=what_test,
            transform=test_transform,
            download=args.is_download,
            compat=True,
        )
        self._test_data_loader = DataLoader(
            self._test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        return
