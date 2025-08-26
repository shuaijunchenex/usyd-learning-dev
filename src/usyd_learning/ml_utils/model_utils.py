import torch
import gc
from .console import console
from torch.optim import Optimizer
from torch import nn

class ModelUtils:

    @staticmethod
    def clear_all(model: nn.modules, optimizer: Optimizer):
        """
        Clears gradients of all models and releases unused cached GPU memory.
        """
        ModelUtils.clear_cuda_cache()
        ModelUtils.clear_model_grads(model)
        ModelUtils.reset_optimizer_state(optimizer)
        console.info("All model grads cleared and cuda cache released.")

    @staticmethod
    def clear_model_grads(model: nn.modules):
        """
        Clears the gradients of all parameters in the given model by setting .grad to None.
        """
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None

        console.info(f"Model grads cleared: {model}, id: {id(model)}")

    @staticmethod
    def clear_cuda_cache():
        """
        Releases unused cached GPU memory to help avoid memory accumulation.
        """
        gc.collect()
        torch.cuda.empty_cache()
        console.info(f"Cuda cache cleared.")

    @staticmethod
    def reset_optimizer_state(optimizer : Optimizer):
        """
        Clears the internal state of an optimizer (e.g., momentum buffers).
        """
        optimizer.state.clear()
        console.info(f"Optimizer state reset: {optimizer}")
