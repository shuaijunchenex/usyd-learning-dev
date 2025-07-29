from __future__ import annotations

from selection.fed_client_select_method import EFedClientSelectMethod
from selection.fed_client_selector_abc import AbstractFedClientSelector

"""
Common implenmentation of client selector
"""


class FedClientSelectorBase(AbstractFedClientSelector):
    def __init__(self, client_list: list, selection_method: EFedClientSelectMethod):
        super().__init__(client_list, selection_method)
        return

    #override
    def _before_select(self, selection_number: int) -> None:
        return

    #override
    def _after_select(self) -> None:
        return
