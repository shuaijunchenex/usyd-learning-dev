from __future__ import annotations

import torch.optim as optim


class OptimizerBuilder:
    """
    A class that automatically constructs a PyTorch optimizer based on the configuration dictionary.

    Args:
        parameters: The model parameters (typically use model.parameters()).
        config_dict: A configuration dictionary (choose either config_path or config_dict).
    """

    def __init__(self, parameters, config_dict):

        self.config = config_dict
        self.parameters = parameters
        self._optimizer: optim.Optimizer = None
        return


    @property
    def optimizer(self):
        """
        optimizer property
        """
        return self._optimizer


    def build(self) -> optim.Optimizer:
        """
        build optimizer return optim.Optimizer
        """

        optimizer_type = self.config.get("type", "sgd").lower()

        #Check lr in dict
        if "lr" not in self.config:
            raise ValueError("Learning rate 'lr' must be specified in optimizer config.")

        kwargs = {"lr": float(self.config.get("lr"))}

        match optimizer_type:
            case "sgd":
                self._optimizer = self.__build_sgd(kwargs)
            case "adam":
                self._optimizer = self.__build_adam(kwargs)
            case "adagrad":
                self._optimizer = self.__build_adagrad(kwargs)
            case "rmsprop":
                self._optimizer = self.__build_rmsprop(kwargs)
            case _:
                self._optimizer = None
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        return self._optimizer


    def rebuild(self, new_parameters) -> optim.Optimizer:
        """
        rebuild optimizer when needed
        """

        self.parameters = new_parameters
        self._optimizer = self.build()
        return self._optimizer

    ##################################################################
    # private functions

    def __safe_add(self, kwargs: dict, key: str, cast_type = None):
        value = self.config.get(key)
        if value is None or str(value).lower() == "none":
            return

        if cast_type is not None:
            try:
                value = cast_type(value)
            except Exception as e:
                raise ValueError(f"Failed to cast optimizer config '{key}' to {cast_type}: {e}")

        kwargs[key] = value
        return


    def __build_sgd(self, kwargs) -> optim.Optimizer:
        """
        SGD
        """

        self.__safe_add(kwargs, "momentum", float)
        self.__safe_add(kwargs, "nesterov", bool)
        self.__safe_add(kwargs, "weight_decay", float)
        return optim.SGD(self.parameters, **kwargs)

    def __build_adam(self, kwargs) -> optim.Optimizer:
        """
        Adam
        """

        self.__safe_add(kwargs, "weight_decay", float)
        self.__safe_add(kwargs, "eps", float)
        self.__safe_add(kwargs, "amsgrad", bool)
        self.__safe_add(kwargs, "betas", lambda v: tuple(map(float, v)))
        return optim.Adam(self.parameters, **kwargs)

    def __build_adagrad(self, kwargs) -> optim.Optimizer:
        """
        Adagrad
        """

        self.__safe_add(kwargs, "weight_decay", float)
        self.__safe_add(kwargs, "eps", float)
        return optim.Adagrad(self.parameters, **kwargs)


    def __build_rmsprop(self, kwargs) -> optim.Optimizer:
        """
        rmsprop
        """

        self.__safe_add(kwargs, "momentum", float)
        self.__safe_add(kwargs, "weight_decay", float)
        self.__safe_add(kwargs, "alpha", float)
        self.__safe_add(kwargs, "centered", bool)
        self.__safe_add(kwargs, "eps", float)
        return optim.RMSprop(self.parameters, **kwargs)
