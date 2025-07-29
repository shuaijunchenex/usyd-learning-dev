from __future__ import annotations
from abc import ABC, abstractmethod

import random

from ...ml_utils import EventHandler


class AbstractFedClientSelector(ABC, EventHandler):
    def __init__(self, random_seed: int = 42):        
        """
        Initialize Selector with client list and select method,
        random seed is set with current time milliseconds(range in 0~999)

        Arg:
            client_list(list): client list to be selected
            selection_method(EFedClientSelectType): select method
        """
        EventHandler.__init__(self)
        self.select_method = None      # Select method
        self_select_number: int = 2
        self_select_round: int = 1

        self._clients_data = None       #Client data
        self.with_random_seed(random_seed)
        return

    def with_random_seed(self, seed: int):
        """
        Manual set random seed
        """
        random.seed(int(seed * 1000) % 1000)
        return self

    def with_clients_data(self, clients_data: any):
        """
        When select clients according to some client data
        """
        self._clients_data = clients_data
        return self

    @abstractmethod
    def select(self, client_list: list, selection_number: int = None):
        """
        Select clients from client list
        """
        pass
