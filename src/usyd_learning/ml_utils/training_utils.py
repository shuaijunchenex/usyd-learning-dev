import random
import numpy as np
import torch

class TrainingUtils:

    @staticmethod
    def set_seed(seed_input: int = 42):
        random.seed(seed_input)
        np.random.seed(seed_input)
        torch.manual_seed(seed_input)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_input)
            torch.cuda.manual_seed_all(seed_input)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
