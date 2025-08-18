import torch
import gc

class ModelUtils:
    @staticmethod
    def clear_model_grads(model):
        """
        Clears the gradients of all parameters in the given model by setting .grad to None.
        """
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None

    @staticmethod
    def clear_cuda_cache():
        """
        Releases unused cached GPU memory to help avoid memory accumulation.
        """
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def reset_optimizer_state(optimizer):
        """
        Clears the internal state of an optimizer (e.g., momentum buffers).
        """
        optimizer.state.clear()
