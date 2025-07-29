import torch
from collections import OrderedDict
from fed_server_aggregation_abc import AbstractFedServerAggregator
from fed_server_aggregation_method import EFedServerAggregationMethod

class Aggregator_RBLA(AbstractFedServerAggregator):

    #TODO: verify if this is the correct import path for the base class

    """
    Implements the RBLA aggregation method.
    """
    
    def __init__(self, aggregation_data_list: list, device: str = "cpu"):
        super().__init__(aggregation_data_list, EFedServerAggregationMethod.fedavg)
        self.device = torch.device(device)

    def aggregate(self, model_weights: list) -> dict:
        """
        Aggregate model weights using FlexLoRA.
        """
        aggregated_weights = {}
        
        for key in model_weights[0].keys():
            aggregated_weights[key] = sum(
                client['weight'] * model_weights[i][key] for i, client in enumerate(self.client_list)
            ) / len(self.client_list)
        
        return aggregated_weights