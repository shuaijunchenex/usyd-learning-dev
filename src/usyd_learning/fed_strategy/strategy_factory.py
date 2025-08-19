from __future__ import annotations
from typing import Callable

from strategy_args import StrategyArgs
from client_strategy import ClientStrategy
from server_strategy import ServerStrategy

class StrategyFactory:
    """
    " Dataset loader factory
    """

    @staticmethod
    def create_args(config_dict: dict, is_clone_dict: bool = False) -> StrategyArgs:
        """
        " Static method to create data loader args
        """
        return StrategyArgs(config_dict, is_clone_dict)

    @staticmethod
    def create_client_strategy(client_strategy_args: StrategyArgs, fn:Callable[[ClientStrategy]]|None = None) -> ClientStrategy:
        """
        " Static method to create data loader
        """
        match client_strategy_args.client_strategy_type.lower():
            case "fedavg":
                from client_strategy_impl._fedavg_client import FedAvgClientTrainingStrategy
                return FedAvgClientTrainingStrategy().create_client_strategy(client_strategy_args, fn)

        raise ValueError(f"Datasdet type '{client_strategy_args.dataset_type}' not support.")

    @staticmethod
    def create_server_strategy(server_strategy_args: StrategyArgs, fn:Callable[[ServerStrategy]]|None = None) -> ServerStrategy:
        """
        " Static method to create server strategy
        """
        match server_strategy_args.server_strategy_type.lower():
            case "fedavg":
                from server_strategy_impl._fedavg_server import FedAvgServerStrategy
                return FedAvgServerStrategy().create_server_strategy(server_strategy_args, fn)

        raise ValueError(f"Server strategy type '{server_strategy_args.server_strategy_type}' not support.")