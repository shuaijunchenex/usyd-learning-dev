from __future__ import annotations
from abc import ABC, abstractmethod

import random
import time
from selection.fed_client_select_method import EFedClientSelectMethod

'''
' Node class interface declare(virtual class as interface)
'''


class AbstractFedClientSelector(ABC):
    def __init__(self, client_list: list, selection_method: EFedClientSelectMethod):        
        """
        Initialize Selector with client list and select method,
        random seed is set with current time milliseconds(range in 0~999)

        Arg:
            client_list(list): client list to be selected
            selection_method(EFedClientSelectType): select method
        """

        #Client list
        self._client_list: list = client_list  
        
        #Select method
        self._selection_method: EFedClientSelectMethod = selection_method

        #Client data
        self._clients_data = None
        
        #Selected clients
        self.selected_clients: list = []

        # seed range milliseconds 0~999
        self.with_random_seed(int(time.time() * 1000) % 1000)
        return


    @property
    def selected_count(self) -> int:
        """
        Selected clients count
        """
        return len(self.selected_clients)


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


    def with_clients_data(self, clients_data: any):
        """
        When select clients according to some client data

        Args:
            client_data: any clients data, maybe dict or list

        Return:
            self
        """

        self._clients_data = clients_data
        return self


    def select(self, selection_number: int):
        """
        Select clients from client list

        Arg:
            select_numbers(int): number of clients to be selected
        """

        self._before_select(selection_number)
        self._do_select(selection_number)
        self._after_select()
        return self.selected_clients

    ########################################
    # Abstract mathod

    @abstractmethod
    def _before_select(self, selection_number: int) -> None:
        """
        Call before select clients
        """
        pass


    @abstractmethod
    def _do_select(self, selection_number: int) -> None:
        """
        do select clients
        """
        pass


    @abstractmethod
    def _after_select(self) -> None:
        """
        Call after select clients
        """
        pass
