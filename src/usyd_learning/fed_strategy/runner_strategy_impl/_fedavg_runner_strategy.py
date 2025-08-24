import tqdm
from ...ml_utils import console
from ...fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from ...fl_algorithms.selection.fed_client_selector_factory import FedClientSelectorFactory
from ...fed_runner import FedRunner
from fed_strategy.runner_strategy import RunnerStrategy 

class FedAvgRunnerStrategy(RunnerStrategy):

    def __init__(self, runner: FedRunner, client_node, server_node) -> None:
        super().__init__(runner, client_node, server_node)

    def create_runner_strategy(self):
        """
        Return a client strategy based on the provided YAML configuration.
        This method is typically called during the node's initialization.
        """
        return self

    def simulate_local_train(self):

        for client in self.client_node:#TODO: modify to iterate client obj
            console.out(f"Client [{client.node_var.node_id}] local training ...")
            updated_weights, train_record = client.run_local_training()
            console.debug(f"Client [{client.node_var.node_id}] local training completed.")
            yield {
                "updated_weights": updated_weights,
                "data_sample_num": len(client.node_var.train_data.dataset),
                "train_record": train_record
            }

    def simulate_server_broadcast(self):
        #TODO

        for client in self.client_node:#TODO: modify to iterate client obj
            # set client weight
            raise NotImplementedError("Subclasses must implement this method.")

        self.runner.server_node.broadcast_weights(self.runner.aggregator.aggregated_weight)
        return

    @staticmethod
    def simulate_server_update(self, weight):
        #TODO
        self._server_node.model_weight = weight
        return

    def run(self) -> None:
        # Implement the FedAvg run logic here
        print("Running FedAvg strategy...")
        for round in tqdm(range(self.runner.training_rounds + 1)):

            client_list = self.runner.client_node_list
            console.out(f"\n{'='*10} Training round {round}/{self.runner.training_rounds}, Total participants: {len(client_list)} {'='*10}")

            self.participants = self.runner.selector.select(self.runner.client_node_list, self.runner.client_node_count)
            console.info(f"Round: {round}, Select {len(self.participants)} clients: ', '").ok(f"{', '.join(map(str, self.participants))}")

            client_updates = self.simulate_local_train()
            
            client_data = []

            for i in client_updates:
               client_data.append([i["updated_weights"], i["data_sample_num"]])

            self.new_aggregated_weight = self._server_node.aggregator.aggregate(client_data)

            #self.server_node.aggregate_weights(client_data) #TODO
            #new_weight = fedavg_aggregator.aggregate_weights(client_data)
            self.simulate_server_update(self.new_aggregated_weight) #self.runner.server_node.update_weights(new_weight)

            self.simulate_server_broadcast() #self.runner.server_node.broadcast_weights(new_weight)

            #Evaluate the global model
            eval_results = self.runner.server_node.evaluate_model(round)

            self.runner.train_logger.record(eval_results)

            console.out(f"{'='*10} Round {round}/{self.runner.training_rounds} End{'='*10}")

            return
        
