from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..fed_strategy.strategy_args import StrategyArgs
from ..ml_utils import TrainingLogger, EventHandler, console, String, ObjectMap, KeyValueArgs

class ServerStrategy(ABC):

    def __init__(self, server_node) -> None:
        self._strategy_type: str = "server"  # 可选：标识具体策略类型
        self._server_node = server_node
        self._args: Optional[StrategyArgs] = None
        self._after_create_fn: Optional[Callable[[ServerStrategy], None]] = None
        self._is_created: bool = False

    # --------------------------------------------------
    @property
    def strategy_type(self) -> str:
        return self._strategy_type

    @property
    def get_server(self):
        return self._server_node

    @property
    def args(self) -> StrategyArgs:
        if self._args is None:
            raise ValueError("ERROR: ServerStrategy's args is None. Call create(...) first.")
        return self._args

    @property
    def is_created(self) -> bool:
        return self._is_created

    # --------------------------------------------------
    def create(
        self,
        args: StrategyArgs,
        fn: Optional[Callable[[ServerStrategy], None]] = None
    ) -> ServerStrategy:
        """
        Create (initialize) the server strategy.
        """
        self._args = args
        self._create_inner(self._server_node.node_var)
        self._is_created = True

        if fn is not None:
            self._after_create_fn = fn
            fn(self)

        return self

    @abstractmethod
    def _create_inner(self, args: KeyValueArgs) -> None:
        """
        Real create logic implemented by subclasses.
        """
        pass

    def create_server_strategy(
        self,
        server_strategy_args: StrategyArgs,
        fn: Optional[Callable[[ServerStrategy], None]] = None
    ) -> ServerStrategy:
        """
        Backward-compatible factory-style entry.
        Equivalent to calling: self.create(server_strategy_args, fn)
        """
        return self.create(server_strategy_args, fn)

    # --------------------------------------------------
    def _assert_created(self) -> None:
        if not self._is_created:
            raise RuntimeError("ServerStrategy is not created. Call create(...) before using it.")

    @abstractmethod
    def aggregation(self) -> dict:
        """
        Aggregate weights from clients.
        :param client_weights: List of weights from clients.
        :return: Aggregated weights.
        """
        pass

    @abstractmethod
    def broadcast(self) -> None:
        """
        Broadcast aggregated weights to clients.
        :param aggregated_weights: The aggregated weights to be broadcast.
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """
        Main loop/step for the strategy (e.g., one FL round orchestration).
        """
        pass

    @abstractmethod
    def evaluate(self) -> None:
        """
        Evaluate server-side performance/metrics.
        """
        pass
