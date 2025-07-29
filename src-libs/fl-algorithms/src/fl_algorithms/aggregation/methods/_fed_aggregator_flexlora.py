import torch
from collections import OrderedDict
from fed_server_aggregation_abc import AbstractFedServerAggregator
from fed_server_aggregation_method import EFedServerAggregationMethod

class _Aggregator_FlexLoRA(AbstractFedServerAggregator):

    #TODO: verify if this is the correct import path for the base class

    """
    Implements the FlexLoRA aggregation method.
    """

    def __init__(self, client_list: list):
        self.client_list = client_list

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