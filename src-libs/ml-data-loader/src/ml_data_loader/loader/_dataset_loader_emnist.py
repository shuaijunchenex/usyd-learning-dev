from __future__ import annotations

from dataset_loader import DatasetLoader
from dataset_loader_args import DatasetLoaderArgs
from dataset_type import EDatasetType

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
Dataset loader for Emnist
'''


class _DatasetLoader_Emnist(DatasetLoader):
    def __init__(self):
        super(_DatasetLoader_Emnist, self).__init__(EDatasetType.emnist)

    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

        dataset = datasets.EMNIST(root = args.root, split = args.split, train=True, transform = args.transform, download = args.is_download)

        self._train_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
        return
