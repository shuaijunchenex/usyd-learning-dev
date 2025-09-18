import random
import numpy as np
import torch

class TrainingUtils:
    
    @staticmethod
    def set_seed(seed: int = 42):
        # Python 内置随机库
        random.seed(seed)
        # NumPy
        np.random.seed(seed)
        # PyTorch (CPU)
        torch.manual_seed(seed)
        # PyTorch (GPU, 单卡)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 多卡

        # 保证 cudnn 使用确定性算法（会牺牲一些性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        np.random.seed(seed)
        # PyTorch (CPU)
        torch.manual_seed(seed)
        # PyTorch (GPU, 单卡)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(torch.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
