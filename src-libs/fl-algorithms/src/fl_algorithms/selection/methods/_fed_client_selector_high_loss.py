from __future__ import annotations

from selection.fed_client_selector_base import EFedClientSelectMethod, FedClientSelectorBase

"""
High loss client selection class
"""


class _FedClientSelector_HighLoss(FedClientSelectorBase):
    def __init__(self, client_list):
        super().__init__(client_list, EFedClientSelectMethod.random)
        return

    #override
    def _before_select(self, selection_number: int) -> None:
        """
        Sort clients based on their average loss in descending order.
        """

        # Convert to list of (client_id, data_pack) and sort by avg_loss descending
        sorted_clients = sorted(self._clients_data.items(),
                              key = lambda item: item[1]["train_record"]["sqrt_train_loss_power_two_sum"],
                              reverse = True)

        # Take top-k
        self.__top_k = [client_id for client_id, _ in sorted_clients[:selection_number]]
        return

    #override
    def _do_select(self, selection_number: int) -> None:
        return [client for client in self._client_list if client.node_id in self.__top_k]
