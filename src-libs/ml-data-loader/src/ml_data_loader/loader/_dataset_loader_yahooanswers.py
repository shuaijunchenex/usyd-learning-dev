from __future__ import annotations

from dataset_loader import DatasetLoader
from dataset_loader_args import DatasetLoaderArgs
from dataset_loader_util import DatasetLoaderUtil
from dataset_type import EDatasetType

from torch.utils.data import DataLoader
from torchtext.datasets import YahooAnswers

'''
Dataset loader for YahooAnswers
'''


class _DatasetLoader_YahooAnswers(DatasetLoader):
    def __init__(self):
        super(_DatasetLoader_YahooAnswers, self).__init__(EDatasetType.dbpedia)

    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        dataset = YahooAnswers(root = args.root, split = args.split)
        self._train_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, collate_fn=DatasetLoaderUtil.text_collate_fn)
        return
