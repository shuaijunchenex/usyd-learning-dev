from __future__ import annotations

from aggregation.fed_server_aggregation_method import EFedServerAggregationMethod
from aggregation.fed_server_aggregation_abc import AbstractFedServerAggregator

from aggregation.methods._fed_aggregator_fedavg import _Aggregator_FedAvg
from aggregation.methods._fed_aggregator_rbla import _Aggregator_RBLA
from aggregation.methods._fed_aggregator_flexlora import _Aggregator_FlexLoRA
'''
' Fed client selector
'''

class FedServerAggregationFactory:

    @staticmethod
    def create_aggregator(aggregation_data_list: list, aggregation_method: EFedServerAggregationMethod) -> AbstractFedServerAggregator:
        match aggregation_method:
            case EFedServerAggregationMethod.fedavg:
                return _Aggregator_FedAvg(aggregation_data_list)
            case EFedServerAggregationMethod.rbla:
                return _Aggregator_RBLA(aggregation_data_list)
            case EFedServerAggregationMethod.flexlora:
                return _Aggregator_FlexLoRA(aggregation_data_list)
            case _:
                raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

