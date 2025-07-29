from __future__ import annotations

import random

from selection.fed_client_selector_base import EFedClientSelectMethod, FedClientSelectorBase

"""
Random clients select class
"""


class _FedClientSelector_Random(FedClientSelectorBase):
    def __init__(self, client_list: list):
        super().__init__(client_list, EFedClientSelectMethod.random)
        return

    # Override parent class virtual method
    def _do_select(self, selection_number: int) -> None:
        self.selected_clients = random.sample(self._client_list, selection_number)
        return