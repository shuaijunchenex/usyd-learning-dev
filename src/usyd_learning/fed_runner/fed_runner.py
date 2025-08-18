from __future__ import annotations
from abc import ABC, abstractmethod

from tqdm import tqdm
import yaml

from ..ml_utils import console
from ..ml_simu_switcher import SimuSwitcher
from ..fed_node import FedNodeClient, FedNodeEdge, FedNodeServer


class FedRunner(ABC):
    def __init__(self):
        self._switcher = SimuSwitcher()          #Simulate net switcher

        self._yaml: dict = {}
        self.training_rounds = 50

        self.client_node_list = []
        self.edge_node_list = []
        self.server_node: FedNodeServer|None = None
        self.run_strategy = None #TODO
        return

    #------------------------------------------
    @property
    def client_node_count(self):
        return len(self.client_node_list)

    @property
    def edge_node_count(self):
        return len(self.edge_node_list)

    def with_switcher(self, switcher):
        self._switcher = switcher
        return self

    def with_yaml(self, runner_yaml):
        self._yaml = runner_yaml
        return self

    #------------------------------------------

    def create_run_strategy(self):
        self.run_strategy = self.__create_run_strategy_from_yaml(self._yaml)

    def __create_run_strategy_from_yaml(self, run_strategy_yaml: str):
        #TODO
        return 

    def create_nodes(self):
        # Create server node(only 1 node)
        self.__create_server_nodes(self._yaml)

        # Create edge nodes
        self.__create_edge_nodes(self._yaml)

        # Create client nodes
        self.__create_client_nodes(self._yaml)
        return

    #private
    def __create_client_nodes(self, runner_yaml: dict):
        # Check 'server_node' if defined
        if "client_nodes" not in runner_yaml:
            console.error_exception("'client_nodes' not defined in node config yaml")

        node_count = 1

        if "client_nodes" in runner_yaml:
            client_section = runner_yaml["client_nodes"]
        else:
            client_section = runner_yaml

        for group in client_section:
            g = client_section[group]
            num: int = g["number"]
            id_prefix = g.get("id_prefix", "")
            link_to = g.get("link_to", "server")

            for index in range(1, num + 1):
                node_id = f"{id_prefix}.{node_count}"
                client = FedNodeClient(node_id, group)

                self.client_node_list.append(client)

                # Create simu node and connect to node
                client.create_simu_node(self._switcher)
                client.connect(link_to)
                node_count += 1

        return

    #private
    def __create_edge_nodes(self, runner_yaml: dict):
        # TODO:
        return

    #private
    def __create_server_nodes(self, runner_yaml: dict):
        # Check 'server_node' if defined
        if "server_node" not in runner_yaml:
            console.warn("'server_node' not defined in runner yaml")
            return

        if "server_node" in runner_yaml:
            server_section = runner_yaml["server_node"]
        else:
            server_section = runner_yaml

        server_id = server_section.get("id", "server")

        self.server_node = FedNodeServer(server_id)

        # Create simu node
        self.server_node.create_simu_node(self._switcher)
        return

    @abstractmethod
    def simulate_local_train():
        """
        Local training simulation method.
        This method should be overridden by subclasses to implement local training logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def simulate_server_broadcast():
        """
        Server broadcast simulation method.
        This method should be overridden by subclasses to implement server broadcast logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def simulate_server_update():
        """
        Server update simulation method.
        This method should be overridden by subclasses to implement server update logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def run(self):
        if self.run_strategy is None:
            raise RuntimeError("run_strategy is not set")
        self.run_strategy.run(self)
