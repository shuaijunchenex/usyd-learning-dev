from __future__ import annotations

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

'''
Dataset loader for Qmnist
'''
class DatasetLoader_Qmnist(DatasetLoader):
    def __init__(self):
        super().__init__()

    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

        self._dataset = datasets.QMNIST(root=args.root, train=True, transform=args.transform, download=args.is_download)
        self._data_loader = DataLoader(self._dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
        return
