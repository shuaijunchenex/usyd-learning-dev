from __future__ import annotations
from typing import Callable

from .strategy_args import StrategyArgs
from .client_strategy import ClientStrategy
from .server_strategy import ServerStrategy

class StrategyFactory:
    """
    " Dataset loader factory
    """

    @staticmethod
    def create_args(config_dict: dict, is_clone_dict: bool = False) -> StrategyArgs:
        """
        " Static method to create data loader args
        """
        #TODO
        return StrategyArgs(config_dict, is_clone_dict)

    @staticmethod
    def create(args: StrategyArgs):
        match args.role.lower():
            case "client":
                return StrategyFactory.create_client_strategy(args)
            case "server":
                return StrategyFactory.create_server_strategy(args)
            case "runner":
                return StrategyFactory.cr

    @staticmethod
    def create_runner_strategy(runner_strategy_args: StrategyArgs, fn:Callable[[ClientStrategy], None] | None = None) -> ClientStrategy:
        """
        " Static method to create runner strategy
        """
        match runner_strategy_args.strategy_name.lower():
            case "fedavg":
                # Import FedAvgRunnerStrategy from the appropriate module
                from usyd_learning.fed_strategy.runner_strategy_impl._fedavg_runner_strategy import FedAvgRunnerStrategy
                return FedAvgRunnerStrategy().create_runner_strategy(runner_strategy_args, fn)

        raise ValueError(f"Runner strategy type '{runner_strategy_args.strategy_name}' not support.")

    @staticmethod
    def create_client_strategy(client_strategy_args: StrategyArgs, fn:Callable[[ClientStrategy]]|None = None) -> ClientStrategy:
        """
        " Static method to create data loader
        """
        match client_strategy_args.strategy_name.lower():
            case "fedavg":
                from client_strategy_impl._fedavg_client import FedAvgClientTrainingStrategy
                return FedAvgClientTrainingStrategy().create_client_strategy(client_strategy_args, fn)

        raise ValueError(f"Client strategy type '{client_strategy_args.strategy_name}' not support.")

    @staticmethod
    def create_server_strategy(server_strategy_args: StrategyArgs, fn:Callable[[ServerStrategy]]|None = None) -> ServerStrategy:
        """
        " Static method to create server strategy
        """
        match server_strategy_args.strategy_name.lower():
            case "fedavg":
                from server_strategy_impl._fedavg_server import FedAvgServerStrategy
                return FedAvgServerStrategy().create_server_strategy(server_strategy_args, fn)

        raise ValueError(f"Server strategy type '{server_strategy_args.strategy_name}' not support.")