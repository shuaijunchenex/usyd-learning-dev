from __future__ import annotations

from aggregation.fed_server_aggregation_method import EFedServerAggregationMethod
from aggregation.fed_server_aggregation_abc import AbstractFedServerAggregator

from aggregation.methods._fed_aggregator_fedavg import Aggregator_FedAvg
from aggregation.methods._fed_aggregator_rbla import Aggregator_RBLA
from aggregation.methods._fed_aggregator_flexlora import Aggregator_FlexLoRA
'''
' Fed client selector
'''

class FedServerAggregationFactory:

    @staticmethod
    def create_aggregator(aggregation_data_list: list, aggregation_method) -> AbstractFedServerAggregator:
        match aggregation_method:
            case "fedavg":
                return Aggregator_FedAvg(aggregation_data_list)
            case "rbla":
                return Aggregator_RBLA(aggregation_data_list)
            case "flexlora":
                return Aggregator_FlexLoRA(aggregation_data_list)
            case _:
                raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

