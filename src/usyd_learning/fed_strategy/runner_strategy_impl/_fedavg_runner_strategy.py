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
            console.out(f"Client [{client.node_id}] local training ...")
            updated_weights, train_record = client.node_var.strategy.run_local_training()
            console.out(f"Client [{client.node_id}] local training completed.")
            yield {
                "updated_weights": updated_weights,
                "train_record": train_record
        }

    def simulate_server_broadcast_process(self):
        for client in self.client_node:#TODO: modify to iterate client obj
            # set client weight
            raise NotImplementedError("Subclasses must implement this method.")

        self.runner.server_node.broadcast_weights(self.runner.aggregator.aggregated_weight)
        return
    
    def simulate_server_update_process(self, weight):
        self._server_node.model_weight = weight
        return

    def run(self) -> None:
        print("Running FedAvg strategy...")
        for round in tqdm(range(self.args.key_value_dict.data['training_rounds'] + 1)):
           
            console.out(f"\n{'='*10} Training round {round}/{self.args.key_value_dict.data['training_rounds']}, Total participants: {len(self.client_nodes)} {'='*10}")
            self.participants = self.server_node.node_var.client_selection.select(self.client_nodes, self.server_node.node_var.config_dict["client_selection"]["number"])
            
            console.info(f"Round: {round}, Select {len(self.participants)} clients: ', '").ok(f"{', '.join(map(str, self.participants))}")

            client_updates = list(self.simulate_client_local_training_process(self.participants))         

            self.new_aggregated_weight = self.server_node.node_var.aggregation_method.aggregate(client_updates)

            self.simulate_server_update(self.new_aggregated_weight) #self.runner.server_node.update_weights(new_weight)

            self.simulate_server_broadcast() #self.runner.server_node.broadcast_weights(new_weight)

            eval_results = self.runner.server_node.strategy.evaluate(round)

            self.runner.train_logger.record(eval_results)

            console.out(f"{'='*10} Round {round}/{self.runner.training_rounds} End{'='*10}")

            return
        
