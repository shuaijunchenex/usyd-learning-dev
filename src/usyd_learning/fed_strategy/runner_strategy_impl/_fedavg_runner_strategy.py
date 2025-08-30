import time

from tqdm import tqdm

from usyd_learning.fed_strategy.strategy_args import StrategyArgs
from ...ml_utils import console
from ...fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from ...fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from ...fed_runner import FedRunner
from ...fed_strategy.runner_strategy import RunnerStrategy 
from ...fed_node import FedNodeClient, FedNodeServer

class FedAvgRunnerStrategy(RunnerStrategy):

    def __init__(self, runner: FedRunner, args, client_node, server_node) -> None:
        super().__init__(runner) #TODO: modify runner object declaration
        self._strategy_type = "fedavg"
        self.args = args
        self.client_nodes : list[FedNodeClient]= client_node
        self.server_node : FedNodeServer = server_node

    def _create_inner(self, client_node, server_node) -> None:
       
        return self

    def simulate_client_local_training_process(self, participants):
        for client in participants:
            console.info(f"[{client.node_id}] Local training started")
            updated_weights, train_record = client.node_var.strategy.run_local_training()
            yield {
                "updated_weights": updated_weights,
                "train_record": train_record
            }
    # def simulate_client_local_training_process(self, participants):
    #     for client in participants:
    #         console.out(f"Client [{client.node_id}] local training start")
    #         updated_weights, train_record = client.node_var.strategy.run_local_training()
    #         console.ok(f"Client [{client.node_id}] local training completed.")
    #         yield {
    #             "updated_weights": updated_weights,
    #             "train_record": train_record
    #     }

    def simulate_server_broadcast_process(self):
        self.server_node.broadcast_weight(self.client_nodes)
        return
    
    def simulate_server_update_process(self, weight):
        self.server_node.node_var.model_weight = weight
        return

    def run(self) -> None:
        print("Running FedAvg strategy...")
        self.server_node.node_var.training_logger.begin()
        for round in tqdm(range(self.args.key_value_dict.data['training_rounds'] + 1)):
           
            console.out(f"\n{'='*10} Training round {round}/{self.args.key_value_dict.data['training_rounds']}, Total participants: {len(self.client_nodes)} {'='*10}")
            self.participants = self.server_node.node_var.client_selection.select(self.client_nodes, self.server_node.node_var.config_dict["client_selection"]["number"])
            
            console.info(f"Round: {round}, Select {len(self.participants)} clients: ', '").ok(f"{', '.join(map(str, self.participants))}")

            client_updates = list(self.simulate_client_local_training_process(self.participants))         

            self.new_aggregated_weight = self.server_node.node_var.aggregation_method.aggregate(client_updates)

            self.simulate_server_update_process(self.new_aggregated_weight)

            self.simulate_server_broadcast_process()

            eval_results = self.server_node.node_var.model_evaluator.evaluate()

            self.server_node.node_var.training_logger.record(eval_results)

            console.out(f"{'='*10} Round {round}/{self.args.key_value_dict.data['training_rounds']} End{'='*10}")

        return
        
