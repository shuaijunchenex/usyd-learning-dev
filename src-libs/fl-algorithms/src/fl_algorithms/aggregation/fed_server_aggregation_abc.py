from __future__ import annotations
from abc import ABC, abstractmethod

import random
import time
from fed_server_aggregation_method import EFedServerAggregationMethod

'''
' Fed Server Aggregator class interface declare
'''


class AbstractFedServerAggregator(ABC):
    def __init__(self, aggregation_data_list: list, aggregation_method: EFedServerAggregationMethod):        
        """
        Initialize aggregator with updated weight list and aggregation method,
        random seed is set with current time milliseconds(range in 0~999)

        Arg:
            aggregation_data_list(list): list of aggregation data, each element is a tuple of (model_weight: dict / wbab, vol)
            aggregation_method(EFedServerAggregationMethod): method to aggregate the weights, such as FedAvg, RBLA, etc.
        """

        #data list
        self._aggregation_data_list: list = aggregation_data_list # [[model_weight: dict / wbab, vol],[model_weight: dict / wbab, vol]]
        
        #Select method
        self._aggregation_method: EFedServerAggregationMethod = aggregation_method

        #Aggregated weight
        self._aggregated_weight: any = None # can be wbab or torch dict or ....

        # seed range milliseconds 0~999
        self.with_random_seed(int(time.time() * 1000) % 1000)
        return


    @property
    def aggregated_weight(self):
        """
        aggregated_weight property
        """

        return self._aggregated_weight


    def with_random_seed(self, seed: int):
        """
        Manual set random seed

        Arg:
            seed(int): random seed

        Return:
            self
        """

        random.seed(seed)
        return self


    def with_clients_update(self, clients_update: any):
        """
        When select clients according to some client data

        Args:
            client_data: any clients data, maybe dict or list

        Return:
            self
        """

        self._clients_update = clients_update
        return self


    def aggregate(self):
        """
        Select clients from client list

        Arg:
            select_numbers(int): number of clients to be selected
        """

        self._before_aggregation()
        self._do_aggregation()
        self._after_aggregation()
        return self._aggregated_weight

    ########################################
    # Abstract method

    @abstractmethod
    def _before_aggregation(self) -> None:
        """
        Call before select clients
        """
        pass


    @abstractmethod
    def _do_aggregation(self) -> None:
        """
        do select clients
        """
        pass


    @abstractmethod
    def _after_aggregation(self) -> None:
        """
        Call after select clients
        """
        pass
