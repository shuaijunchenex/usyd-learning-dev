import random
import numpy as np
import torch
import contextlib

class TrainingUtils:

    @staticmethod
    def make_autocast(device: torch.device, enabled: bool):
        if not enabled:
            return contextlib.nullcontext()
        if device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=torch.float16)
        if device.type == "mps":
            return torch.autocast(device_type="mps", dtype=torch.float16)
        
        return contextlib.nullcontext()

    @staticmethod
    def make_scaler(device: torch.device, enabled: bool):
        if enabled and device.type == "cuda":
            return torch.cuda.amp.GradScaler(enabled=True)
        return None

    @staticmethod
    def set_seed_all(seed_input: int = 42):
        random.seed(seed_input)
        np.random.seed(seed_input)
        torch.manual_seed(seed_input)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_input)
            torch.cuda.manual_seed_all(seed_input)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def set_seed(seed_input: int = 42):
        random.seed(seed_input)
        np.random.seed(seed_input)
        torch.manual_seed(seed_input)