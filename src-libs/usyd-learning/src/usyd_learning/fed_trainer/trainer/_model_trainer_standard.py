import torch
import math

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..model_trainer_args import ModelTrainerArgs
from ..model_trainer import ModelTrainer

from ...ml_algorithms import ModelExtractor
from ...ml_utils import console


class ModelTrainer_Standard(ModelTrainer):
    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)

        if str(next(trainer_args.model.parameters()).device) != trainer_args.device:
            self.model = trainer_args.model.to(trainer_args.device)
        else:
            self.model = trainer_args.model
        return

    def train_step(self):
        self.model.train()
        running_loss = 0.0
        loop = tqdm(self.trainer_args.train_loader, desc="Training", leave=True, ncols=100, mininterval=0.1, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]")
        total_batch = 0

        for inputs, labels in loop:
            total_batch += 1
            inputs, labels = inputs.to(self.trainer_args.device), labels.to(self.trainer_args.device)

            self.trainer_args.optimizer.zero_grad()
            outputs = self.trainer_args.model(inputs)
            loss = self.trainer_args.loss_func(outputs, labels)
            loss.backward()
            self.trainer_args.optimizer.step()

            running_loss += loss.item()
            #loop.set_postfix(loss=loss.item())

        return running_loss / total_batch

    def train(self, epochs, is_return_wbab = False):
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
        # console.info(f"\n[Summary] Total Loss: {train_stats['train_loss_sum']:.4f} | Avg Loss: {train_stats["avg_loss"]:.4f}")

        # # Optional: comment out or reduce the debug prints for model params
        # console.info("\n[Debug] Model param means:")
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         console.info(f"  {name}: {param.data.mean():.6f}")

        if is_return_wbab == False:
            return self.model.state_dict(), train_stats
        else:
            return self.model.state_dict(), train_stats, self.extract_wbab()
    
    def observe(self, epochs=5):
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
        Extracts the model parameters using the AdvancedModelExtractor.
        """
        
        return ModelExtractor(self.model).get_layer_dict()
    