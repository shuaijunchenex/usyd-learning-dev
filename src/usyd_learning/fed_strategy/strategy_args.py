from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, Callable, Union
from ..ml_utils.key_value_args import KeyValueArgs

Object = str  # "client" | "server"
StrategyOwner = Union[object]  # 这里 owner 既可能是 client 也可能是 server/fednode

@dataclass
class StrategyArgs(KeyValueArgs):
    """
    统一的策略参数，兼容 client/server。
    关键字段：
      - role: "client" 或 "server"（区分目标对象）
      - strategy_name:   策略名（用于工厂注册/检索）
    其余字段保持可扩展，与 DatasetLoaderArgs 一致使用 KeyValueArgs
    """
    role: str = "client"                     # "client" | "server"
    strategy_name: str = "fedavg"                       # 策略名
    observation_epochs: int = 0                 # 仅对 client 有意义的示例字段
    local_epochs: int = 1                       # 仅对 client 有意义的示例字段
    optimizer: Dict[str, Any] = field(default_factory=dict)
    trainer: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, config_dict: Optional[dict] = None, is_clone_dict: bool = False):
        KeyValueArgs.__init__(self, config_dict, is_clone_dict)
        self.role = self.get("role", self.role).lower()
        if self.role not in ("client", "server"):
            raise ValueError(f"[StrategyArgs] 'role' must be 'client' or 'server', got: {self.role}")
        self.strategy_name = self.get("strategy_name", self.strategy_name)

    def set_server_strategy_args(self):
        # client_selection:
        # method: random
        # round: 1
        # number: 2
        # random_seed: 42
        self.aggregation_args = self.get("model_aggregation")
        self.client_selection_args = self.get("client_selection")

    def set_client_strategy_args(self):
        self.observation_epochs = int(self.get("observation_epochs", self.observation_epochs))
        self.local_epochs = int(self.get("local_epochs", self.local_epochs))
        self.optimizer = dict(self.get("optimizer", self.optimizer))
        self.trainer = dict(self.get("trainer", self.trainer))
        self.seed = self.get("seed", self.seed)
        self.extra = dict(self.get("extra", self.extra))

    def client_strategy_as_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "strategy_name": self.strategy_name,
            "observation_epochs": self.observation_epochs,
            "local_epochs": self.local_epochs,
            "optimizer": self.optimizer,
            "trainer": self.trainer,
            "seed": self.seed,
            "extra": self.extra,
        }
    # """
    # Strategy type
    # """
    # strategy_obj: str = "client" # or "server"

    # """
    # strategy mode
    # """
    # strategy_type: str = "fedavg" # "fedavg", "fedprox", "scaffold", etc.

    # def __init__(self, config_dict: dict|None = None, is_clone_dict = False):
    #     super().__init__(config_dict, is_clone_dict)

    #     if config_dict is not None and "strategy" in config_dict:
    #         self.set_args(config_dict["strategy"], is_clone_dict)

    #     self.strategy_type = self.get("type", "fedavg")
    #     self.strategy_obj = self.get("role", None) # can be server or client

    #     return
