import copy

from ..client_strategy import ClientStrategy
from ...ml_utils.model_utils import ModelUtils
from ...model_trainer import model_trainer_factory

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

        optimizer = self._build_optimizer(trainable_params, self.config) #TODO: check

        # Initialize the model trainer
        self.trainer = model_trainer_factory.create(self.config) #TODO
        
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

        optimizer = self._build_optimizer(trainable_params, self.config) #TODO: change config to optimizer

        # Initialize the model trainer
        self.trainer = model_trainer_factory.create(self.config)
        
        # Call the trainer for local training
        updated_weights, train_record = self.trainer.train(self.client.args.local_epochs)

        # Update model weights
        self.client.update_weights(updated_weights)

        self.client.args.local_model.load_state_dict(updated_weights)

        return copy.deepcopy(updated_weights), train_record