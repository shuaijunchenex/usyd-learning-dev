"""
Entry class of Sample_1
"""

from usyd_learning.fed_node import FedNodeVars, FedNodeEventArgs
from usyd_learning.fed_runner import FedRunner
from usyd_learning.ml_utils import AppEntry, console

class SampleAppEntry(AppEntry):
    def __init__(self):
        super().__init__()    

        # Define runner, client, server, edge yaml variables, can be set outside manually
        self.runner_yaml = None
        self.client_yaml = None 
        self.edge_yaml = None 
        self.server_yaml = None 

    #override
    def run(self, training_rounds = 50):

        # Yamls - if yamls are None, get yaml from app config file automatically
        if self.runner_yaml is None:
            self.runner_yaml = self.get_app_object("runner")
        if self.client_yaml is None:
            self.client_yaml = self.get_app_object("client_yaml")
        if self.edge_yaml is None:
            self.edge_yaml = self.get_app_object("edge_yaml")
        if self.server_yaml is None:
            self.server_yaml = self.get_app_object("server_yaml")

        # Training rounds
        self.fed_runner = FedRunner()       # Runner
        self.fed_runner.training_rounds = training_rounds
        self.fed_runner.with_yaml(self.runner_yaml)
        self.fed_runner.create_nodes()
        self.fed_runner.create_run_strategy()

        # Prepare each client node and var
        client_var_list = []
        for index, node in enumerate(self.fed_runner.client_node_list):
            client_var = FedNodeVars(self.client_yaml)
            client_var.prepare() #TODO: create client strategy
            self.__attach_event_handler(client_var)

            # Two way binding
            client_var.owner_nodes = node
            node.node_var = client_var
            client_var.prepare_strategy_only()
            client_var_list.append(client_var)

        # Prepare each edge node and var
        # edge_var_list = []
        # for index in range(self.fed_runner.edge_node_count):
        #     edge_var = FedNodeVars(self.edge_yaml)
        #     #edge_var.prepare()
        #     self.__attach_event_handler(edge_var)

        #     # Two way bind
        #     edge_var.owner_nodes = node
        #     node.node_var = edge_var
        #     edge_var_list.append(edge_var)

        # Prepare server node and var
        server_var = FedNodeVars(self.server_yaml)
        server_var.prepare() #TODO: create server strategy
        self.__attach_event_handler(server_var)
        server_var.owner_nodes = self.fed_runner.server_node        # Two way binding
        server_var.prepare_strategy_only()
        self.fed_runner.server_node.node_var = server_var

        self.fed_runner.run()
        
        return

    # Attach events to node variable object
    def __attach_event_handler(self, node_var):
        node_var.attach_event("on_prepare_data_loader", self.on_prepare_data_loader)
        node_var.attach_event("on_prepare_model", self.on_prepare_model)
        node_var.attach_event("on_prepare_loss_func", self.on_prepare_loss_func)
        node_var.attach_event("on_prepare_optimizer", self.on_prepare_optimizer)      #attach EVENT
        node_var.attach_event("on_prepare_strategy", self.on_prepare_strategy)    
        node_var.attach_event("on_prepare_extractor", self.on_prepare_extractor)
        node_var.attach_event("on_prepare_model", self.on_prepare_model)
        node_var.attach_event("on_prepare_loss_func", self.on_prepare_loss_func)
        node_var.attach_event("on_prepare_data_distribution", self.on_prepare_data_distribution)
        node_var.attach_event("on_prepare_data_handler", self.on_prepare_data_handler)
        node_var.attach_event("on_prepare_client_selection", self.on_prepare_client_selection)
        node_var.attach_event("on_prepare_trainer", self.on_prepare_trainer)
        node_var.attach_event("on_prepare_aggregation", self.on_prepare_aggregation)
        node_var.attach_event("on_prepare_extractor", self.on_prepare_extractor)
        node_var.attach_event("on_prepare_strategy", self.on_prepare_strategy)
        node_var.attach_event("on_prepare_training_logger", self.on_prepare_training_logger)        

    # Event handdlers
    # region
    def on_prepare_data_loader(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_data_loader event")
        return

    def on_prepare_model(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_model event")
        return

    def on_prepare_optimizer(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_optimizer event")
        return

    def on_prepare_loss_func(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_loss_func event")
        return

    def on_prepare_data_distribution(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_data_distribution event")
        return

    def on_prepare_data_handler(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_data_handler event")
        return

    def on_prepare_client_selection(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_client_selection event")
        return

    def on_prepare_trainer(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_training event")
        return

    def on_prepare_aggregation(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_aggregation event")
        return

    def on_prepare_strategy(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_strategy event")
        return

    def on_prepare_extractor(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_extractor event")
        return

    def on_prepare_training_logger(self, args: FedNodeEventArgs):
        console.warn(f"TODO: on_prepare_training_logger event")
        return
    # endregion
