from __future__ import annotations

from dataset.dataset_loader import DatasetLoader
from dataset.dataset_loader_args import DatasetLoaderArgs
from dataset.dataset_type import EDatasetType

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
Dataset loader for ImageNet
'''


class DatasetLoader_ImageNet(DatasetLoader):
    def __init__(self):
        super(DatasetLoader_ImageNet, self).__init__(EDatasetType.emnist)

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        '''
        继承虚方法实现
        '''
        if args.transform is None:
            args.transform = transforms.Compose([   
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        dataset = datasets.ImageNet(root= args.root, split = args.split, transform = args.transform)
        self._train_data_loader = DataLoader(dataset, args.batch_size, args.shuffle, args.num_workers)
        return

    def _after_create(self, args: DatasetLoaderArgs) -> None:
        '''
        继承虚方法实现
        '''
        return