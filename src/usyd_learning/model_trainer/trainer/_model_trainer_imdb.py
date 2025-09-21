import math
import contextlib
from typing import Any
import torch
import torch.nn as nn

from torch.cuda.amp import GradScaler  # 仅在 CUDA + AMP 时实际启用
from usyd_learning.ml_utils.model_utils import ModelUtils
from usyd_learning.model_trainer import ModelTrainer, ModelTrainerArgs

# 你已提供：
# from usyd_learning.ml_utils.training_utils import TrainingUtils   # 含 make_autocast
# from usyd_learning.ml_utils.ewma import ModelEWMA                 # 或按你的路径导入
from usyd_learning.ml_utils.training_utils import TrainingUtils
from usyd_learning.ml_utils.model_ewma import ModelEWMA


class ModelTrainer_Imdb(ModelTrainer):
    """
    Drop-in replacement of ModelTrainer_Standard:
    - 复用 TrainingUtils.make_autocast(device, enabled) 作为 AMP 上下文
    - 复用 ModelEWMA 维护 EMA 权重（可选）
    - 其它接口/行为与原版一致（不配置即退化为原始逻辑）
    """

    def __init__(self, trainer_args: ModelTrainerArgs):
        super().__init__(trainer_args)

        if trainer_args.model is None:
            raise ValueError("Training Model is None.")
        if trainer_args.optimizer is None:
            raise ValueError("Training optimizer is None.")

        self.device = ModelUtils.accelerator_device()
        self.model: nn.Module = trainer_args.model

        # ==== 新增：可选 AMP / EMA ====
        ta = self.trainer_args
        self.amp_enabled: bool = bool(getattr(ta, "amp_enabled", False))

        # GradScaler：仅在 CUDA + AMP 时启用，可通过 use_grad_scaler=False 关闭
        self.use_grad_scaler: bool = bool(getattr(ta, "use_grad_scaler", True))
        self._scaler = None
        if self.amp_enabled and torch.cuda.is_available() and self.use_grad_scaler:
            self._scaler = GradScaler(enabled=True)

        # EMA：传入 ema_decay ∈ (0,1) 时启用
        ema_decay = getattr(ta, "ema_decay", None)
        self._ema = None
        if isinstance(ema_decay, (float, int)) and 0.0 < float(ema_decay) < 1.0:
            self._ema = ModelEWMA(self.model, decay=float(ema_decay), device=self.device)

        # 确保模型在目标设备
        if str(next(self.model.parameters()).device) != str(self.trainer_args.device):
            self.model = self.model.to(self.trainer_args.device)
            self.trainer_args.model = self.model

    # ---- 接口保持不变 ----
    def set_model(self, model: nn.Module):
        self.trainer_args.model = model
        if str(next(model.parameters()).device) != self.trainer_args.device:
            self.trainer_args.model = model.to(self.trainer_args.device)
        self.model = self.trainer_args.model
        return self

    # ---- 内部小上下文：与原逻辑等价，但可扩展 ----
    def _ctx_model_train(self):
        self.trainer_args.model.train()
        return contextlib.nullcontext()

    def train_step(self) -> float:
        ta = self.trainer_args
        if ta.optimizer is None:
            raise ValueError("Trainer optimizer is None.")
        if ta.model is None:
            raise ValueError("Trainer model is None.")
        if ta.loss_func is None:
            raise ValueError("Trainer loss function is None.")
        if ta.train_loader is None:
            raise ValueError("Trainer train_loader is None.")

        train_dl = ta.train_loader.data_loader
        if not hasattr(train_dl, "__iter__"):
            raise TypeError(f"train_loader must be an iterable DataLoader, got {type(train_dl).__name__}")

        total_epochs = getattr(ta, "total_epochs", getattr(ta, "epochs", None))

        ta.model.to(self.device)
        running_loss, total_batch = 0.0, 0

        from tqdm.auto import tqdm
        loop = tqdm(
            train_dl,
            desc=f"Training (epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''})",
            leave=True, ncols=120, mininterval=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        # 训练模式上下文
        with self._ctx_model_train():
            for inputs, labels in loop:
                total_batch += 1
                inputs = inputs.to(ta.device)
                labels = labels.to(ta.device)

                ta.optimizer.zero_grad(set_to_none=True)

                # 你的 autocast 上下文（根据 device 与 enabled 自动选择/退化）
                with TrainingUtils.make_autocast(device=self.device, enabled=self.amp_enabled):
                    outputs = ta.model(inputs)
                    loss = ta.loss_func(outputs, labels)

                # 反传（带/不带 GradScaler）
                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                    self._scaler.step(ta.optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    ta.optimizer.step()

                # EMA（可选）
                if self._ema is not None:
                    self._ema.update(ta.model)

                running_loss += float(loss.detach().item())
                loop.set_postfix(
                    batch=total_batch,
                    loss=f"{float(loss):.4f}",
                    avg_loss=f"{running_loss/total_batch:.4f}",
                    lr=ta.optimizer.param_groups[0]["lr"]
                )

        avg_loss = running_loss / max(total_batch, 1)

        from tqdm.auto import tqdm as _tqdm
        _tqdm.write(
            f"[Epoch {self._epoch_idx}{'/' + str(total_epochs) if total_epochs else ''} Finished] "
            f"avg_loss={avg_loss:.6f} | batches={total_batch} | device={ta.device}"
        )
        return avg_loss

    def train(self, epochs, is_return_wbab=False) -> Any:
        self.trainer_args.total_epochs = epochs
        self._epoch_idx = 0

        stats = {"train_loss_sum": 0, "epoch_loss": [], "train_loss_power_two_sum": 0}
        for _ in range(epochs):
            self._epoch_idx += 1
            loss = self.train_step()
            stats["train_loss_sum"] += loss
            stats["train_loss_power_two_sum"] += loss ** 2
            stats["epoch_loss"].append(loss)

        self._epoch_idx = 0
        stats["avg_loss"] = stats["train_loss_sum"] / max(epochs, 1)
        stats["sqrt_train_loss_power_two_sum"] = math.sqrt(stats["train_loss_power_two_sum"])

        return self.trainer_args.model.state_dict(), stats

    def observe(self, epochs=5) -> Any:
        self.trainer_args.total_epochs = epochs
        stats = {"train_loss_sum": 0, "epoch_loss": [], "train_loss_power_two_sum": 0}

        for _ in range(epochs):
            loss = self.train_step()
            stats["train_loss_sum"] += loss
            stats["train_loss_power_two_sum"] += loss ** 2
            stats["epoch_loss"].append(loss)

        stats["avg_loss"] = stats["train_loss_sum"] / max(epochs, 1)
        stats["sqrt_train_loss_power_two_sum"] = math.sqrt(stats["train_loss_power_two_sum"])
        return self.trainer_args.model.state_dict(), stats
