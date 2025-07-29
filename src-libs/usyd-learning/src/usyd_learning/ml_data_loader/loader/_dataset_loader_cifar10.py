from __future__ import annotations

from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
Dataset loader for Cifar10
'''
class DatasetLoader_Cifar10(DatasetLoader):
    def __init__(self):
        super().__init__()

    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        dataset = datasets.CIFAR10(root = args.root, train = True, transform = args.transform, download = args.is_download)
        self.train_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

        args.is_load_train_set = True
        args.is_load_test_set = False

        return
