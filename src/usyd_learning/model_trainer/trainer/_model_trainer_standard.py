from typing import Any
import torch.nn as nn
import math
from tqdm import tqdm

from ..model_trainer_args import ModelTrainerArgs
from ..model_trainer import ModelTrainer

from ...ml_algorithms import ModelExtractor
from ...ml_utils import console


class ModelTrainer_Standard(ModelTrainer):
    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)

        if trainer_args.model is None:
            raise ValueError("Training Model is None.")
        
        if trainer_args.optimizer is None:
            raise ValueError("Training optimizer is None.")

        if str(next(trainer_args.model.parameters()).device) != trainer_args.device:
            self.model: nn.Module = trainer_args.model.to(trainer_args.device)
        else:
            self.model: nn.Module = trainer_args.model
        return

    # override
    def train_step(self) -> float:
        if self.trainer_args.optimizer is None: 
            raise ValueError("Trainer optimizer is None.")
        if self.trainer_args.model is None:                 
            raise ValueError("Trainer model is None")
        
        self.model.train()
        running_loss = 0.0
        loop = tqdm(self.trainer_args.train_loader, desc="Training", leave=True, ncols=100, mininterval=0.1, 
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]")
        total_batch = 0

        for inputs, labels in loop:
            total_batch += 1
            inputs = inputs.to(self.trainer_args.device)
            labels = labels.to(self.trainer_args.device)

            self.trainer_args.optimizer.zero_grad()
            outputs = self.trainer_args.model(inputs)
            loss = self.trainer_args.loss_func(outputs, labels)
            loss.backward()
            self.trainer_args.optimizer.step()

            running_loss += loss.item()
            #loop.set_postfix(loss=loss.item())

        return running_loss / total_batch

    # override
    def train(self, epochs, is_return_wbab = False) -> Any:
        train_stats = {"train_loss_sum": 0, "epoch_loss": [], "train_loss_power_two_sum":0}

        console.info(f"\nTraining Start ({epochs} epochs)")
        for epoch in range(epochs):
            train_loss = self.train_step()
            train_stats["train_loss_sum"] += train_loss
            train_stats["train_loss_power_two_sum"] += train_loss ** 2
            train_stats["epoch_loss"].append(train_loss)
            # console.debug(f"Epoch {epoch + 1:02d}/{epochs} - Loss: {train_loss:.4f}")

        train_stats["avg_loss"] = train_stats["train_loss_sum"] / epochs
        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["train_loss_power_two_sum"])

        if is_return_wbab == False:
            return self.model.state_dict(), train_stats
        else:
            return self.model.state_dict(), train_stats, self.extract_wbab()
    
    def observe(self, epochs=5) -> Any:
        train_stats = {"train_loss_sum": 0, "epoch_loss": [], "train_loss_power_two_sum":0}

        console.info(f"\nObservation start ({epochs} epochs)")
        for epoch in range(epochs):
            train_loss = self.train_step()
            train_stats["train_loss_sum"] += train_loss
            train_stats["train_loss_power_two_sum"] += train_loss ** 2
            train_stats["epoch_loss"].append(train_loss)
            # console.info(f"Epoch {epoch + 1:02d}/{epochs} - Loss: {train_loss:.4f}")

        train_stats["avg_loss"] = train_stats["train_loss_sum"] / epochs
        train_stats["sqrt_train_loss_power_two_sum"] = math.sqrt(train_stats["train_loss_power_two_sum"])
        # console.info(f"\n[Summary] Total Loss: {train_stats['train_loss_sum']:.4f} | Avg Loss: {train_stats["avg_loss"]:.4f}")

        return self.model.state_dict(), train_stats

    def extract_wbab(self):
        """
        Extracts the model parameters using the ModelExtractor.
        """        
        return ModelExtractor().extract_layers(self.model)
    