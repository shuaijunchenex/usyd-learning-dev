import copy
import torch
from typing import Any, Tuple

from usyd_learning.fed_strategy.strategy_args import StrategyArgs

from ..client_strategy import ClientStrategy
from ...ml_utils.model_utils import ModelUtils
from ...model_trainer import model_trainer_factory
from ...model_trainer.model_trainer_args import ModelTrainerArgs
from ...model_trainer.model_trainer_factory import ModelTrainerFactory
from ...ml_algorithms.optimizer_builder import OptimizerBuilder
from ...ml_data_loader import DatasetLoaderFactory
from ...ml_algorithms.loss_function_builder import LossFunctionBuilder
from ...ml_utils import console
from ...fed_node.fed_node_vars import FedNodeVars

import copy
from typing import Any, Tuple
import torch
import torch.nn as nn

class FedAvgClientTrainingStrategy(ClientStrategy):
    def __init__(self):
        """
        client: a FedNodeClient (or FedNode) that owns a FedNodeVars in `client.node_var`
        config: high-level strategy/trainer config; falls back to `client.node_var.config_dict` when needed
        """
        super().__init__()

    def _create_inner(self, args, client_node) -> None:
        self._args = args
        self._strategy_type = "fedavg"
        self._obj = client_node
        return

    # ------------------- Public: Observation wrapper -------------------
    def run_observation(self) -> dict:
        print(f"\n Observation Client [{self._obj.node_var.node_id}] ...\n")
        updated_weights, train_record = self.observation()
        return {
            "node_id": self._obj.node_var.node_id,
            "train_record": train_record,
            "data_sample_num": self._dataset_size(self._obj.node_var.node_var), # TODO: update to sample num
        }

    # ------------------- Observation (no state write-back) -------------------
    def observation(self) -> Tuple[dict, Any]:
        """
        Lightweight local training using the current node_var model weights.
        Return updated weights and training log, but DO NOT write back to node state.
        """
        node_vars: FedNodeVars = self._obj.node_var.node_var
        if node_vars is None:
            raise RuntimeError("client.node_var is not set. Use client.with_node_var(...) first.")

        # 1) Copy model and load current weights (acts like 'global' init for this client)
        observe_model: nn.Module = copy.deepcopy(node_vars.model)
        if node_vars.model_weight is not None:
            observe_model.load_state_dict(node_vars.model_weight, strict=True)

        # 2) Resolve config/device and build optimizer/loss/trainer via base helpers
        # Prefer explicit `self.config`; fall back to node_vars.config_dict
        full_cfg = self.config or node_vars.config_dict or {}
        device = node_vars.device if hasattr(node_vars, "device") and node_vars.device else "cpu"

        ModelUtils.clear_cuda_cache(device)
        console.log(f"Cuda cache cleared: {device}")
        ModelUtils.clear_model_grads(observe_model)
        console.log(f"Model grads cleared: {observe_model}")

        optimizer = self._build_optimizer(self, observe_model, full_cfg)
        ModelUtils.reset_optimizer_state(optimizer)
        console.log(f"Optimizer state reset: {optimizer}")

        loss_cfg = full_cfg.get("loss", full_cfg)
        loss_func = LossFunctionBuilder.build(loss_cfg)

        trainer_type = full_cfg.get("trainer", {}).get("trainer_type", "standard")
        save_path = full_cfg.get("trainer", {}).get("save_path", None)

        train_loader = self._get_torch_dataloader(node_vars)

        self.trainer = self._build_trainer(
            model=observe_model,
            optimizer=optimizer,
            loss_func=loss_func,
            train_loader=train_loader,
            device=device,
            trainer_type=trainer_type,
            save_path=save_path,
        )

        # 3) Train for local epochs (use observe() if your concrete trainer provides it)
        local_epochs = int(full_cfg.get("training", {}).get("local_epochs", 1))
        updated_weights, train_record = self.trainer.train(local_epochs)

        return copy.deepcopy(updated_weights), train_record

    # ------------------- Public: Local training wrapper -------------------
    def run_local_training(self) -> dict:
        print(f"\n Training Client [{self._obj.node_var.node_id}] ...\n")
        updated_weights, train_record = self.local_training()
        return {
            "node_id": self._obj.node_var.node_id,
            "updated_weights": updated_weights,
            "train_record": train_record,
            "data_sample_num": self._obj.node_var.data_sample_num, #TODO: update
        }

    # ------------------- Full local training (write-back to node_var) -------------------
    def local_training(self) -> Tuple[dict, Any]:
        """
        Full local training: initialize from node_vars.model_weight, train, then write updated weights
        back to node_vars.model_weight and node_vars.model.
        """
        node_vars: FedNodeVars = self._obj.node_var
        if node_vars is None:
            raise RuntimeError("client.node_var is not set. Use client.with_node_var(...) first.")

        # 1) Sync cached model with current node_var.model_weight (acts like global init)
        if node_vars.model_weight is not None:
            node_vars.model.load_state_dict(node_vars.model_weight)

        # 2) Train on a copied model to keep node_vars.model intact during the run
        train_model: nn.Module = copy.deepcopy(node_vars.model)

        # 3) Build optimizer, loss, and trainer
        full_cfg = self.config or node_vars.config_dict or {}
        device = node_vars.device if hasattr(node_vars, "device") and node_vars.device else "cpu"

        ModelUtils.clear_cuda_cache(device)
        console.log(f"Cuda cache cleared: {device}")
        ModelUtils.clear_model_grads(train_model)
        console.log(f"Model grads cleared: {train_model}")

        optimizer = self._build_optimizer(self, train_model, full_cfg)
        ModelUtils.reset_optimizer_state(optimizer)
        console.log(f"Optimizer state reset: {optimizer}")

        loss_cfg = full_cfg.get("loss", full_cfg)
        loss_func = LossFunctionBuilder.build(loss_cfg)

        trainer_type = full_cfg.get("trainer", {}).get("trainer_type", "standard")
        save_path = full_cfg.get("trainer", {}).get("save_path", None)

        train_loader = self._get_torch_dataloader(node_vars)

        self.trainer = self._obj._build_trainer(
            model=train_model,
            optimizer=optimizer,
            loss_func=loss_func,
            train_loader=train_loader,
            device=device,
            trainer_type=trainer_type,
            save_path=save_path,
        )

        # 4) Train for local epochs
        local_epochs = int(full_cfg.get("training", {}).get("local_epochs", 1))
        updated_weights, train_record = self.trainer.train(local_epochs)

        # 5) Write-back: update node_var weights and sync into its model
        node_vars.model_weight = copy.deepcopy(updated_weights)
        node_vars.model.load_state_dict(node_vars.model_weight, strict=True)

        # Optional: if your FedNodeClient exposes update_weights(), keep this too
        if hasattr(self.client, "update_weights"):
            self.client.update_weights(node_vars.model_weight)

        return copy.deepcopy(updated_weights), train_record
