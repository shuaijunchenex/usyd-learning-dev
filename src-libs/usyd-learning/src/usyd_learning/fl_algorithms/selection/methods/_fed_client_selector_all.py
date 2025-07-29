from __future__ import annotations

from ..fed_client_selector_abc import AbstractFedClientSelector

"""
Select all clients
"""

class FedClientSelector_All(AbstractFedClientSelector):
    def __init__(self, random_seed: int = 42):
        super().__init__(random_seed)
        self.select_method = "all"      # Select method     
        return

    #Override parent class virtual method
    def select(self, client_list: list, selection_number: int = None):
        """
        Select clients from client list
        """        
        return client_list