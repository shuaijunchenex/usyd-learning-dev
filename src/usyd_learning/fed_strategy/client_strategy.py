from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

from model_trainer import ModelTrainer, ModelTrainerFactory, ModelTrainerArgs
from ml_algorithms import OptimizerBuilder

try:
    from ..ml_utils import console
except Exception:
    console = None


class ClientStrategy(ABC):
    """Abstract base for a client's local-training/observation strategy."""

    def __init__(self, client, config: Optional[dict] = None) -> None:
        self._client = client
        self._args: Optional[dict] = None           
        self._config: Optional[dict] = config       
        self.trainer: Optional[ModelTrainer] = None

        self._after_create_fn: Optional[Callable[[ClientStrategy], None]] = None
        self._is_created: bool = False

    # -------------------------- properties --------------------------
    @property
    def client(self):
        return self._client

    @property
    def args(self) -> dict:
        if self._args is None:
            raise ValueError("ClientStrategy.args is None. Call create(...) first.")
        return self._args

    @property
    def config(self) -> Optional[dict]:
        return self._config

    @property
    def is_created(self) -> bool:
        return self._is_created

    # -------------------------- lifecycle --------------------------
    def create(
        self,
        args: dict,
        fn: Optional[Callable[[ClientStrategy], None]] = None
    ) -> ClientStrategy:
        """
        Initialize the strategy instance with given args, then run optional after-create callback.
        """
        self._args = args
        self._create_inner(args)
        self._is_created = True

        if fn is not None:
            self._after_create_fn = fn
            fn(self)

        return self

    # 向后兼容：保持你原有的入口名称
    def create_client_strategy(
        self,
        client_strategy_args: dict,
        fn: Optional[Callable[[ClientStrategy], None]] = None
    ) -> ClientStrategy:
        return self.create(client_strategy_args, fn)

    @abstractmethod
    def _create_inner(self, args: dict) -> None:
        """Subclass should implement real construction here."""
        raise NotImplementedError

    # -------------------------- high-level wrappers --------------------------
    def run_observation(self) -> Dict[str, Any]:
        """
        High-level wrapper: do a lightweight observation step and package the result.
        """
        self._assert_created()
        updated_weights, info = self.observation()
        pack = {
            "node_id": getattr(self.client, "node_id", None),
            "updated_weights": updated_weights,
            "record": info,
            "mode": "observation",
        }
        return pack

    def run_local_training(self) -> Dict[str, Any]:
        """
        High-level wrapper: do a full local training step and package the result.
        """
        self._assert_created()
        updated_weights, train_record = self.local_training()

        data_num = None
        try:
            # 尽量兼容你的 client.args.train_data.dataset 写法
            data_num = len(self.client.args.train_data.dataset)
        except Exception:
            pass

        pack = {
            "node_id": getattr(self.client, "node_id", None),
            "updated_weights": updated_weights,
            "train_record": train_record,
            "data_sample_num": data_num,
            "mode": "local_training",
        }
        return pack

    # -------------------------- abstract core steps --------------------------
    @abstractmethod
    def observation(self) -> Tuple[Dict[str, Any], Any]:
        """Lightweight training using current global weights (no client state update)."


# # client_strategy.py
# from __future__ import annotations
# from abc import ABC, abstractmethod
# from typing import Any, Tuple
# import copy
# import torch

# from ..model_trainer.model_trainer_args import ModelTrainerArgs
# from ..model_trainer.model_trainer import ModelTrainer
# from ..model_trainer.model_trainer_factory import ModelTrainerFactory
# from ..ml_algorithms.optimizer_builder import OptimizerBuilder
# from ..ml_utils import console

# class ClientStrategy(ABC):
#     Abstract base for a client's local-training/observation strategy."""

#     def __init__(self, client, config) -> None:
#         self.client = client
#         self.trainer: ModelTrainer | None = None
#         self.config = config

#     def create_client_strategy(self, client_strategy_args, fn):
#         """
#         Return a client strategy based on the provided YAML configuration.
#         This method is typically called during the node's initialization.
#         """
#         self.client_strategy_args = client_strategy_args
#         self.fn = fn

#         return self

#     # -------- high-level wrappers --------
#     def run_observation(self) -> dict:
#         pass

#     def run_local_training(self) -> dict:
#         pass

#     @abstractmethod
#     def observation(self) -> Tuple[dict, Any]:
#         """Lightweight training using the current global weights (no client state update)."""
#         raise NotImplementedError

#     @abstractmethod
#     def local_training(self) -> Tuple[dict, Any]:
#         """Full local training; should update the client's local weights."""
#         raise NotImplementedError

#     # -------- helpers --------
#     @staticmethod
#     def _build_optimizer(self, model, config):
#         """Build an optimizer for the model."""
#         optimizer = OptimizerBuilder(model.parameters(), config_dict = config["optimizer"]).build()
#         console.info(f"Optimizer created from configuration dictionary:\n{optimizer}")
#         return optimizer

#     def _build_trainer(
#         self,
#         *,
#         model,
#         optimizer,
#         loss_func,
#         train_loader,
#         device: str = "cpu",
#         trainer_type: str = "standard",
#         save_path: str | None = None,
#         extra: dict | None = None,
#     ) -> ModelTrainer:
#         """Create a concrete ModelTrainer via the factory."""
#         cfg = {
#             "trainer": {
#                 "trainer_type": trainer_type,
#                 "model": model,
#                 "optimizer": optimizer,
#                 "loss_func": loss_func,
#                 "train_loader": train_loader,
#                 "device": device,
#             }
#         }
#         if save_path:
#             cfg["trainer"]["save_path"] = save_path
#         if extra:
#             cfg["trainer"].update(extra)

#         args: ModelTrainerArgs = ModelTrainerFactory.create_args(cfg)
#         trainer: ModelTrainer = ModelTrainerFactory.create(args)
#         return trainer
