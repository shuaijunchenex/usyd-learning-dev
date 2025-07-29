import copy

from train_strategy.client_strategy.base_client_strategy import ClientStrategy
from trainer.standard_model_trainer import StandardModelTrainer
from tools.optimizer_builder import OptimizerBuilder
from tools.model_utils import ModelUtils

class FedAvgClientTrainingStrategy(ClientStrategy):
    def __init__(self, client):
        self.client = client

    def run_observation(self):

        print(f"\n Observation Client [{self.client.node_id}] ...\n")

        updated_weights, train_record = self.observation()

        data_pack = {"node_id": self.client.node_id, "train_record": train_record, "data_sample_num": len(self.client.args.train_data.dataset)}

        return data_pack

    def observation(self):
        '''
        For light-weight client observation training, we use the local LoRA model.
        '''

        # clear gradients
        # ModelUtils.clear_model_grads(self.client.args.local_model)

        observe_model = copy.deepcopy(self.client.args.local_model)

        # Retrieve the current model weights
        current_weight = self.client.args.global_weight

        # Set global weight
        observe_model.load_state_dict(current_weight)

        trainable_params = [p for p in observe_model.parameters() if p.requires_grad]

        optimizer = OptimizerBuilder(trainable_params, self.client.args.optimizer).optimizer

        # Initialize the model trainer
        self.trainer = StandardModelTrainer(observe_model,
                                    optimizer,
                                    self.client.args.loss_func,
                                    self.client.args.train_data)
        
        # Call the trainer for local training
        updated_weights, train_record = self.trainer.train(int(self.client.args.local_epochs))

        return copy.deepcopy(updated_weights), train_record

    def run_local_training(self):

        print(f"\n Training Client [{self.client.node_id}] ...\n")

        updated_weights, train_record = self.local_training()

        data_pack = {"node_id": self.client.node_id, "updated_weights": updated_weights, "train_record": train_record, "data_sample_num": len(self.client.args.train_data.dataset)}

        return data_pack

    def local_training(self):
        # Correct
        # Set global weight
        self.client.args.local_model.load_state_dict(self.client.args.global_weight)

        train_model = copy.deepcopy(self.client.args.local_model)

        # clear gradients
        ModelUtils.clear_model_grads(train_model)

        trainable_params = [p for p in train_model.parameters() if p.requires_grad]

        optimizer = OptimizerBuilder(trainable_params, self.client.args.optimizer).optimizer

        # Initialize the model trainer
        self.trainer = StandardModelTrainer(train_model,#self.client.args.local_model,
                                    optimizer, #torch.optim.SGD(trainable_params, lr=0.01),
                                    self.client.args.loss_func,
                                    self.client.args.train_data)
        
        # Call the trainer for local training
        updated_weights, train_record = self.trainer.train(self.client.args.local_epochs)

        # Update model weights
        self.client.update_weights(updated_weights)

        self.client.args.local_model.load_state_dict(updated_weights)

        return copy.deepcopy(updated_weights), train_record

    # # for debug optimizer only
    # def local_training(self):
    #     # Correct
    #     # Set global weight
    #     self.client.args.local_model.load_state_dict(self.client.args.global_weight)

    #     train_model = copy.deepcopy(self.client.args.local_model)

    #     # clear gradients
    #     ModelUtils.clear_model_grads(train_model)

    #     import torch.nn as nn
    #     import torch.optim as optim

    #     # 3. SGD 优化器
    #     optimizer = optim.SGD(train_model.parameters(), lr=0.05, momentum=0)

    #     # Initialize the model trainer
    #     self.trainer = ModelTrainer(train_model,#self.client.args.local_model,
    #                                 optimizer, #torch.optim.SGD(trainable_params, lr=0.01),
    #                                 self.client.args.loss_func,
    #                                 self.client.args.train_data)
        
    #     # Call the trainer for local training
    #     updated_weights, train_record = self.trainer.train(self.client.args.local_epochs)

    #     # Update model weights
    #     self.client.update_weights(updated_weights)

    #     self.client.args.local_model.load_state_dict(updated_weights)

    #     return copy.deepcopy(updated_weights), train_record
