import torch
import torch.nn as nn

class LossFunctionBuilder:
    """
    A class to build a PyTorch loss function from a configuration dictionary.

    Args:
        config_dict: A dictionary containing loss function configuration.

        loss_func:
            type: CrossEntropyLoss  # Options: CrossEntropyLoss, MSELoss, etc.
            reduction: mean         # Options: mean, sum, none
            weight: None            # Optional: Define class weights for imbalanced datasets
    """

    @staticmethod
    def build(config: dict):
        loss_type = config.get("type", "CrossEntropyLoss")
        reduction = config.get("reduction", "mean")
        weight = config.get("weight", None)

        # Handle optional class weights
        if weight is not None:
            weight = torch.tensor(weight, dtype = torch.float)

        kwargs = {"reduction": reduction}
        if weight is not None:
            kwargs["weight"] = weight

        # Build loss function
        if loss_type.lower() == "crossentropyloss":
            return nn.CrossEntropyLoss(**kwargs)
        elif loss_type.lower() == "mseloss":
            return nn.MSELoss(**kwargs)
        elif loss_type.lower() == "l1loss":
            return nn.L1Loss(**kwargs)
        elif loss_type.lower() == "nllloss":
            return nn.NLLLoss(**kwargs)
        else:
            raise ValueError(f"Unsupported loss function type: {loss_type}")