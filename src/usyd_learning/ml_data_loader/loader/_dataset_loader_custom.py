from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from torch.utils.data import DataLoader, Dataset

class DatasetLoader_Custom(DatasetLoader):
    """
    Custom dataset loader.
    """
    def __init__(self):
        super().__init__()

    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        """
        Create DataLoader(s) from a custom Dataset provided in args.dataset.
        """
        
        self._dataset = args.dataset

        self.data_sample_num = len(self._dataset) 
        self.task_type = args.task_type

        self._data_loader = DataLoader(
            self._dataset,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            collate_fn=args.collate_fn
        )

        return
