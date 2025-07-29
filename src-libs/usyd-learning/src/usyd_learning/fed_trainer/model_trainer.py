from abc import ABC, abstractmethod

from .model_trainer_args import ModelTrainerArgs

"""
Model trainer abstract base class
"""

class ModelTrainer(ABC):
    def __init__(self, trainer_args: ModelTrainerArgs):
        self.trainer_args = trainer_args

    @abstractmethod
    def train_step(self):
        """
        Performs a single training step.
        """
        pass

    @abstractmethod
    def train(self, epochs):
        """
        Trains the model for a number of epochs.
        """
        pass

    def observe(self, epochs = 5):
        """
        Performs observation without updating the global state.
        """
        pass

    def extract_WbAB(self):
        """
        Extracts structured model components (e.g., LoRA components).
        """
        pass
