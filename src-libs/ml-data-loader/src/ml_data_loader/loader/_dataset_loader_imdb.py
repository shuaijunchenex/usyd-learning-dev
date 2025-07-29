from __future__ import annotations

from dataset_loader import DatasetLoader
from dataset_loader_args import DatasetLoaderArgs
from dataset_loader_util import DatasetLoaderUtil
from dataset_type import EDatasetType

from torch.utils.data import DataLoader
from torchtext.datasets import IMDB

'''
Dataset loader for imdb
'''


class _DatasetLoader_Imdb(DatasetLoader):
    def __init__(self):
        super(_DatasetLoader_Imdb, self).__init__(EDatasetType.imdb)

    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        dataset = IMDB(root = args.root, split = args.split)
        self._train_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, collate_fn = DatasetLoaderUtil.text_collate_fn)
        return
