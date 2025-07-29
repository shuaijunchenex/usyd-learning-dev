from __future__ import annotations

from selection.fed_client_selector_base import EFedClientSelectMethod, FedClientSelectorBase

"""
Select all clients
"""


class _FedClientSelector_All(FedClientSelectorBase):
    def __init__(self, client_list: list):
        super().__init__(client_list, EFedClientSelectMethod.all)
        return


    #Override parent class virtual method
    def _do_select(self, selection_number: int) -> None:
        self.selected_clients = self._client_list
        return