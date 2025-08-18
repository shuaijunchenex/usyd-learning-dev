from fed_strategy.server_strategy import ServerStrategy
from model_trainer.model_evaluator import ModelEvaluator
from tqdm import tqdm

class BaseFedAvgServerStrategy(ServerStrategy):

    #TODO: add test dataloader
    def __init__(self, server):
        super().__init__(server)
        self.model_evaluator = ModelEvaluator(server.node_var.model(), server.node_var.data_loader(), server.node_var.loss_func(), server.node_var.device)

    def aggregation(self, client_weights):
        return super().aggregation(client_weights)
    
    def broadcast(self, aggregated_weights):
        return super().broadcast(aggregated_weights)

    def run(self):
        print(f"\n🚀 Training Client [{self.client.node_id}] ...\n")

        updated_weights, train_record = self.local_training()

        data_pack = {"node_id": self.client.node_id, "updated_weights": updated_weights, "train_record": train_record, "data_sample_num": len(self.client.args.train_data.dataset)}

        return data_pack
    
    def evaluate(self):
        self.model_evaluator.evaluate()
        self.model_evaluator.print_results()
        return