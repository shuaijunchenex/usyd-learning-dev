from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import gc

"""
Common NN Model Utils
"""

class ModelUtils:

    @staticmethod
    def clear_model_grads(model : nn.Model):
        """
        Clears the gradients of all parameters in the given model by setting .grad to None.
        """

        for param in model.parameters():
            if param.grad is not None:
                param.grad = None
        return

    @staticmethod
    def clear_cuda_cache():
        """
        Releases unused cached GPU memory to help avoid memory accumulation.
        """

        gc.collect()
        torch.cuda.empty_cache()
        return

    @staticmethod
    def reset_optimizer_state(optimizer: optim):
        """
        Clears the internal state of an optimizer (e.g., momentum buffers).
        """

        optimizer.state.clear()
        return
