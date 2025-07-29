from __future__ import annotations

from ..fed_client_selector_abc import AbstractFedClientSelector

"""
High loss client selection class
"""

class FedClientSelector_HighLoss(AbstractFedClientSelector):
    def __init__(self, random_seed: int = 42):
        super().__init__(random_seed)
        self.select_method = "high_loss"      # Select method     
        return

    #Override parent class virtual method
    def select(self, client_list: list, selection_number: int = None):
        """
        Select clients from client list
        """
        # Convert to list of (client_id, data_pack) and sort by avg_loss descending
        sorted_clients = sorted(self._clients_data.items(),
                              key = lambda item: item[1]["train_record"]["sqrt_train_loss_power_two_sum"],
                              reverse = True)

        # Take top-k
        self.__top_k = [client_id for client_id, _ in sorted_clients[:selection_number]]
        return [client for client in client_list if client.node_id in self.__top_k]
