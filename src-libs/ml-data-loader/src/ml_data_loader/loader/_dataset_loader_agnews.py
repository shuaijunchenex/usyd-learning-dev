from __future__ import annotations

from ml_data_loader.dataset_loader import DatasetLoader
from ml_data_loader.dataset_loader_args import DatasetLoaderArgs
from ml_data_loader.dataset_loader_util import DatasetLoaderUtil
from ml_data_loader.dataset_type import EDatasetType

import torch
from torch.utils.data import DataLoader
from torchtext.datasets import AG_NEWS

'''
Dataset loader for agnews
'''


class _DatasetLoader_Agnews(DatasetLoader):
    def __init__(self):
        super(_DatasetLoader_Agnews, self).__init__(EDatasetType.agnews)

    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        dataset = AG_NEWS(root = args.root, split = args.split)
        self._train_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, collate_fn = DatasetLoaderUtil.text_collate_fn)
        return
