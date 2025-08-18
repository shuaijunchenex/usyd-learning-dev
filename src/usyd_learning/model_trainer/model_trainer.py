from abc import ABC, abstractmethod
from typing import Any

from .model_trainer_args import ModelTrainerArgs


class ModelTrainer(ABC):
    """
    Model trainer abstract base class
    """

    def __init__(self, trainer_args: ModelTrainerArgs):
        self.trainer_args: ModelTrainerArgs = trainer_args

    @abstractmethod
    def train_step(self) -> float:
        """
        Performs a single training step.
        """
        pass

    @abstractmethod
    def train(self, epochs, is_return_wbab = False) -> Any:
        """
        Trains the model for a number of epochs.
        """
        pass

    def set_optimizer(self, optimizer):
        """
        Sets the optimizer for the trainer.
        """
        self.trainer_args.optimizer = optimizer

    def observe(self, epochs=5) -> Any:
        """
        Performs observation without updating the global state.
        """
        pass

    def extract_wbab(self):
        """
        Extracts structured model components (e.g., LoRA components).
        """
        pass
