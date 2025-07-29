from __future__ import annotations

from dataset_loader import DatasetLoader
from dataset_loader_args import DatasetLoaderArgs
from dataset_type import EDatasetType

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
Dataset loader for Kmnist
'''


class _DatasetLoader_Kmnist(DatasetLoader):
    def __init__(self):
        super(_DatasetLoader_Kmnist, self).__init__(EDatasetType.kmnist)

    #override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        if args.transform is None:
            args.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])

        dataset = datasets.KMNIST(root= args.root, train= True, transform= args.transform, download= args.is_download)
        self._train_data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
        return
