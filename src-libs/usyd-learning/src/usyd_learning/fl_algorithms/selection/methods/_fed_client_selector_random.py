from __future__ import annotations

import random

from ..fed_client_selector_abc import AbstractFedClientSelector

"""
Random clients select class
"""


class FedClientSelector_Random(AbstractFedClientSelector):
    def __init__(self, client_list: list):
        super().__init__(client_list)
        self.select_method = "random"
        return

    def select(self, client_list: list, select_number: int = None):
        if select_number is None:
            select_number = self.select_number

        return random.sample(client_list, select_number)
