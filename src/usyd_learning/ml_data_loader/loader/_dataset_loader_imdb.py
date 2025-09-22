from __future__ import annotations


from ..dataset_loader import DatasetLoader
from ..dataset_loader_args import DatasetLoaderArgs
from ..dataset_loader_util import DatasetLoaderUtil
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torch.utils.data import IterableDataset
from ...ml_algorithms.tokenlizer_builder import TokenizerBuilder
from functools import partial

import torch
'''
Dataset loader for imdb
'''
class DatasetLoader_Imdb(DatasetLoader):
    def __init__(self):
        super().__init__()

    def _warmup_download(self, root: str):
        for sp in ("train", "test"):
            it = iter(IMDB(root=root, split=sp))
            try:
                next(it)
            except StopIteration:
                pass

    # override
    def _create_inner(self, args: DatasetLoaderArgs) -> None:
        """
        目标：对齐 MNIST 的接口输出，保证：
          - self._dataset / self._data_loader：训练集（或用户指定 split）
          - self._test_dataset / self._test_data_loader：测试集
          - DataLoader 迭代产出 (Tensor, Tensor)，其中 labels 是 shape [B] 的 long tensor
          - 文本必须通过 text_collate_fn 做 tokenize + pad，保证 batch 内同长、可被 torch.cat
        """
        root = getattr(args, "root")
        is_download = getattr(args, "is_download", True)
        batch_size = getattr(args, "batch_size", 32)
        test_batch_size = getattr(args, "test_batch_size", None) or batch_size
        shuffle = getattr(args, "shuffle", True)
        num_workers = getattr(args, "num_workers", 0)
        
        self._dataset = IMDB(root=root, split="train")
        self._test_dataset = IMDB(root=root, split="test")

        self.vocab = TokenizerBuilder.build_vocab(self._dataset, args.tokenizer)
        args.vocab_size = len(self.vocab)

        if is_download:
            self._warmup_download(root)

        self._data_loader = DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle= False,
            num_workers=num_workers,
            collate_fn=partial(DatasetLoaderUtil.text_collate_fn, tokenizer=args.tokenizer, vocab=self.vocab),
        )

        self.data_sample_num = 25000
        self.task_type = "nlp"
        self._test_data_loader = DataLoader(
            self._test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=partial(DatasetLoaderUtil.text_collate_fn, tokenizer=args.tokenizer, vocab=self.vocab),
        )
        return

    def get_dataset(self) -> DataLoader:
        if self._data_loader is not None:
            return self._data_loader.dataset
        else:
            raise ValueError("ERROR: DatasetLoader's data_loader is None.")
