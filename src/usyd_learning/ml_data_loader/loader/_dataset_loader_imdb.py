from __future__ import annotations

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

from torch.utils.data import DataLoader
from torchtext.datasets import IMDB

'''
Dataset loader for imdb
'''
class DatasetLoader_Imdb(DatasetLoader):
    def __init__(self):
        super().__init__()

    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        self._dataset = IMDB(root = args.root, split = args.split)
        self._data_loader = DataLoader(self._dataset, batch_size=args.batch_size, 
                                       shuffle=args.shuffle, num_workers=args.num_workers, 
                                       collate_fn = args.text_collate_fn)
        return
