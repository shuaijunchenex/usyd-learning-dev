from train_strategy.server_strategy.base_server_strategy import ServerStrategy

class FedAvgServerTrainingStrategy(ServerStrategy):
    def run(self):
        print(f"\n🚀 Training Client [{self.client.node_id}] ...\n")

        updated_weights, train_record = self.local_training()

        data_pack = {"node_id": self.client.node_id, "updated_weights": updated_weights, "train_record": train_record, "data_sample_num": len(self.client.args.train_data.dataset)}

        return data_pack
    
    def local_training(self):
        # Retrieve the current model weights
        current_weight = self.client.args.global_weight

        # Set global weight
        self.client.args.global_model.load_state_dict(current_weight)

        # Call the trainer for local training
        updated_weights, train_record = self.client.trainer.train(self.client.args.local_epochs)

        # Update model weights
        self.client.update_weights(updated_weights)

        return updated_weights, train_record