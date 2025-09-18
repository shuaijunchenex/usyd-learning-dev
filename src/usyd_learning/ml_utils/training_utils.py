import random
import numpy as np
import torch

class TrainingUtils:

    @staticmethod
    def set_seed(seed_input: int = 42):
        # Python 内置随机库
        random.seed(seed_input)
        # NumPy
        np.random.seed(seed_input)
        # PyTorch (CPU)
        torch.manual_seed(seed_input)
        # PyTorch (GPU, 单卡)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_input)
            torch.cuda.manual_seed_all(seed_input)  # 多卡

        # 保证 cudnn 使用确定性算法（会牺牲一些性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        np.random.seed(seed_input)
        # PyTorch (CPU)
        torch.manual_seed(seed_input)
        # PyTorch (GPU, 单卡)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_input)
            
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
