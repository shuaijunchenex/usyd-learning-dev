from node.inode import AbstractNode
from trainer.model_evaluator import ModelEvaluator
from model_adaptor.lora_model_weight_adaptor import LoRAModelWeightAdapter
from model_extractor.advanced_model_extractor import AdvancedModelExtractor 

import copy

class ServerNode(AbstractNode):
    def __init__(self, node_id, server_args):
        # The server node only needs to call common initialization, no training data is required.
        super().__init__(node_id)
        self.args = server_args
        self.evaluator = ModelEvaluator(self.args.global_model, self.args.test_data)
        self.extractor = AdvancedModelExtractor(self.args.global_model)
        self.global_model_WbAB = self.extractor.get_layer_dict()

    def run(self, **kwargs):
        """Server node executes distribution or aggregation operations."""
        return self.distribute()

    def distribute(self):
        # Example: Implement server-side distribution logic
        print(f"Server Node {self.node_id} is distributing the global model.")
        # Return distribution results or updated state as needed.
        # This is just an example.
        return None

    def evaluate_global_model(self, round):
        print("Server Evaluation Started...\n") 

        evaluation_dict = self.evaluator.evaluate()
        evaluation_dict = {"round": round, **evaluation_dict}
        self.evaluator.print_results()

        print("Server Evaluation Completed.\n")

        return evaluation_dict

    def assign_client(self, client_list):
        self.client_list = client_list

    def update_weights(self, new_weight):
        self.args.global_weight = copy.deepcopy(new_weight)
        self.args.global_model.load_state_dict(self.args.global_weight)

    def update_WbAB(self, new_WbAB):
        LoRAModelWeightAdapter.apply_weights_to_model(self.args.global_model, new_WbAB)
        # self.extractor.model = t
        self.global_model_WbAB = self.extractor.get_layer_dict()
        self.evaluator.model = self.args.global_model

        # for name, param in self.args.global_model.named_parameters():
        #     print(f"{name}: mean={param.data.mean().item():.6f}")

    def create_node_info(self):
        info = super().create_node_info()
        # Add server-specific information
        info.update({
            'role': 'server'
        })
        return info