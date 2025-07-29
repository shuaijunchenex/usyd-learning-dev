from __future__ import annotations

from dataset.dataset_loader import DatasetLoader
from dataset.dataset_loader_args import DatasetLoaderArgs
from dataset.dataset_type import EDatasetType

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
Dataset loader for Kmnist
'''


class DatasetLoader_Kmnist(DatasetLoader):
    def __init__(self):
        super(DatasetLoader_Kmnist, self).__init__(EDatasetType.kmnist)

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        '''
        继承虚方法实现
        '''
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])

        dataset = datasets.KMNIST(root= args.root, train= True, transform= args.transform, download= args.is_download)
        self._train_data_loader = DataLoader(dataset, args.batch_size, args.shuffle, args.num_workers)
        return

    def _after_create(self, args: DatasetLoaderArgs) -> None:
        '''
        继承虚方法实现
        '''
        return