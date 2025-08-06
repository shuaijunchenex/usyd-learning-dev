from __future__ import annotations

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil

import torch
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS

"""
Dataset loader for agnews
"""


class DatasetLoader_Agnews(DatasetLoader):
    def __init__(self):
        super().__init__()

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        dataset = AG_NEWS(root=args.root, split=args.split)
        self.train_data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            collate_fn=args.text_collate_fn)

        args.is_load_train_set = True
        args.is_load_test_set = False
        return
