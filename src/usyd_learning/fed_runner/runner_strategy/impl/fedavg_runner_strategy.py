import tqdm
from ....ml_utils import console
from ....fl_algorithms.aggregation.fed_aggregator_facotry import FedAggregatorFactory
from ...fed_runner import FedRunner

from ..runner_strategy import RunnerStrategy 

class FedAvgRunner(RunnerStrategy):
    def run(self) -> None:
        # Implement the FedAvg run logic here
        print("Running FedAvg strategy...")
        for round in tqdm(range(self.training_rounds + 1)):
            # TODO
            # client_list = []
            # console.out(f"\n{'='*10} Training round {round}/{self.training_rounds}, Total participants: {len(self.client_node_list)} {'='*10}")

            # client_selection = self.server_node.node_var.client_selection
            # participants = client_selection.select(self.client_node_list)
            # console.info(f"Round: {round}, Select {len(participants)} clients: ', '").ok(f"{', '.join(map(str, participants))}")

            # client_updates = self._local_train_simu(participants)
            
            # client_data = []

            # for i in client_updates:
            #    client_data.append([i["updated_weights"], i["data_sample_num"]])

            # self.server_node.aggregate_weights(client_data) #TODO
            # new_weight = fedavg_aggregator.aggregate_weights(client_data)
            # server.update_weights(new_weight)

            # for client in client_list:
            #    client.update_weights(new_weight)

            # Evaluate the global model
            # eval_results = server.evaluate_global_model(round)

            # logger.record(eval_results)

            # console.out(f"{'='*10} Round {round}/{self.training_rounds} End{'='*10}")

            return