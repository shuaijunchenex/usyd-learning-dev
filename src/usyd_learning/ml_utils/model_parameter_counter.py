from . import console


class ModelParameterCounter:
    """
    A utility class to count the number of parameters in a PyTorch model.
    """
    
    @staticmethod
    def count_parameters(model):
        """
        Count total and trainable parameters in a PyTorch model.

        Args:
            model (nn.Module): The model to analyze.

        Returns:
            total_params (int): Total number of parameters.
            trainable_params (int): Number of trainable (requires_grad=True) parameters.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        console.info(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")  

        return total_params, trainable_params