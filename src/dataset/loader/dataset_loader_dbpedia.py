from __future__ import annotations

from dataset.dataset_loader import DatasetLoader
from dataset.dataset_loader_args import DatasetLoaderArgs
from dataset.dataset_loader_util import DatasetLoaderUtil
from dataset.dataset_type import EDatasetType

from torch.utils.data import DataLoader
from torchtext.datasets import DBpedia

'''
Dataset loader for DBpedia
'''


class DatasetLoader_DBpedia(DatasetLoader):
    def __init__(self):
        super(DatasetLoader_DBpedia, self).__init__(EDatasetType.dbpedia)

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        '''
        继承虚方法实现
        '''    
        dataset = DBpedia(root = args.root, split = args.split)
        self._train_data_loader = DataLoader(dataset, args.batch_size, args.shuffle, args.num_workers, collate_fn = DatasetLoaderUtil.text_collate_fn)
        return


    def _after_create(self, args: DatasetLoaderArgs) -> None:
        '''
        继承虚方法实现
        '''
        return