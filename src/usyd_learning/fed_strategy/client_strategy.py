# client_strategy.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Tuple
import copy
import torch

from ..model_trainer.model_trainer_args import ModelTrainerArgs
from ..model_trainer.model_trainer import ModelTrainer
from ..model_trainer.model_trainer_factory import ModelTrainerFactory
from ..ml_algorithms.optimizer_builder import OptimizerBuilder
from ..ml_utils import console

class ClientStrategy(ABC):
    """Abstract base for a client's local-training/observation strategy."""

    def __init__(self, client, config) -> None:
        self.client = client
        self.trainer: ModelTrainer | None = None
        self.config = config

    # -------- high-level wrappers --------
    def run_observation(self) -> dict:

        pass
        # """Run lightweight observation (no global update)."""
        # print(f"\n Observation Client [{self.client.node_id}] ...\n")
        # updated_weights, train_record = self.observation()
        # return {
        #     "node_id": self.client.node_id,
        #     "train_record": train_record,
        #     "data_sample_num": len(self.client.args.train_data.dataset),
        # }

    def run_local_training(self) -> dict:

        pass
        # """Run local training and return updated weights + stats."""
        # print(f"\n Training Client [{self.client.node_id}] ...\n")
        # updated_weights, train_record = self.local_training()
        # return {
        #     "node_id": self.client.node_id,
        #     "updated_weights": updated_weights,
        #     "train_record": train_record,
        #     "data_sample_num": len(self.client.args.train_data.dataset),
        # }

    @abstractmethod
    def observation(self) -> Tuple[dict, Any]:
        """Lightweight training using the current global weights (no client state update)."""
        raise NotImplementedError

    @abstractmethod
    def local_training(self) -> Tuple[dict, Any]:
        """Full local training; should update the client's local weights."""
        raise NotImplementedError

    # -------- helpers --------
    @staticmethod
    def _build_optimizer(self, model, config):
        """Build an optimizer for the model."""
        optimizer = OptimizerBuilder(model.parameters(), config_dict = config["optimizer"]).build()
        console.info(f"Optimizer created from configuration dictionary:\n{optimizer}")
        return optimizer

    def _build_trainer(
        self,
        *,
        model,
        optimizer,
        loss_func,
        train_loader,
        device: str = "cpu",
        trainer_type: str = "standard",
        save_path: str | None = None,
        extra: dict | None = None,
    ) -> ModelTrainer:
        """Create a concrete ModelTrainer via the factory."""
        cfg = {
            "trainer": {
                "trainer_type": trainer_type,
                "model": model,
                "optimizer": optimizer,
                "loss_func": loss_func,
                "train_loader": train_loader,
                "device": device,
            }
        }
        if save_path:
            cfg["trainer"]["save_path"] = save_path
        if extra:
            cfg["trainer"].update(extra)

        args: ModelTrainerArgs = ModelTrainerFactory.create_args(cfg)
        trainer: ModelTrainer = ModelTrainerFactory.create(args)
        return trainer
