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
        return StrategyArgs(config_dict, is_clone_dict)

    @staticmethod
    def create(args: StrategyArgs, node):
        match args.role.lower():
            case "client":
                return StrategyFactory.create_client_strategy(args, node)
            case "server":
                return StrategyFactory.create_server_strategy(args, node)

    @staticmethod
    def create_runner_strategy(runner_strategy_args: StrategyArgs, client_nodes, server_node) -> ClientStrategy:
        """
        " Static method to create runner strategy
        """
        match runner_strategy_args.strategy_name.lower():
            case "fedavg":
                # Import FedAvgRunnerStrategy from the appropriate module
                from usyd_learning.fed_strategy.runner_strategy_impl._fedavg_runner_strategy import FedAvgRunnerStrategy
                return FedAvgRunnerStrategy(runner_strategy_args, client_nodes, server_node)

        raise ValueError(f"Runner strategy type '{runner_strategy_args.strategy_name}' not support.")

    @staticmethod
    def create_client_strategy(client_strategy_args: StrategyArgs, client_node_input) -> ClientStrategy:
        """
        " Static method to create data loader
        """
        match client_strategy_args.strategy_name.lower():
            case "fedavg":
                from usyd_learning.fed_strategy.client_strategy_impl._fedavg_client import FedAvgClientTrainingStrategy
                return FedAvgClientTrainingStrategy(client_strategy_args, client_node_input)

        raise ValueError(f"Client strategy type '{client_strategy_args.strategy_name}' not support.")

    @staticmethod
    def create_server_strategy(server_strategy_args: StrategyArgs, serve_node_input) -> ServerStrategy:
        """
        " Static method to create server strategy
        """
        match server_strategy_args.strategy_name.lower():
            case "fedavg":
                from usyd_learning.fed_strategy.server_strategy_impl._fedavg_server import FedAvgServerStrategy
                return FedAvgServerStrategy(server_strategy_args, serve_node_input)

        raise ValueError(f"Server strategy type '{server_strategy_args.strategy_name}' not support.")