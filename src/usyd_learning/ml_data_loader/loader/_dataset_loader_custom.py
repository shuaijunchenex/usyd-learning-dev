from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from torch.utils.data import DataLoader, Dataset

class DatasetLoader_Custom(DatasetLoader):
    """
    Custom dataset loader.
    用于加载用户提供的自定义 Dataset (例如 non-IID 客户端数据集)
    """
    def __init__(self):
        super().__init__()

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        """
        Create DataLoader(s) from a custom Dataset provided in args.dataset.
        """
        
        # 保存 dataset
        self._dataset = args.dataset

        # 主训练 DataLoader
        self._data_loader = DataLoader(
            self._dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            collate_fn=args.collate_fn
        )

        # test_data_loader（如果 dataset 提供 test split，可以传 Dataset；否则 None）
        # 在多数情况下，custom dataset 没有 test split，就设 None
        self._test_data_loader = None
        return
