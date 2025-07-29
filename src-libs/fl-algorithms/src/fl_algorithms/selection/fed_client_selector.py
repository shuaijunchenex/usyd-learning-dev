from __future__ import annotations

from selection.fed_client_select_method import EFedClientSelectMethod
from selection.methods._fed_client_selector_all import _FedClientSelector_All
from selection.methods._fed_client_selector_high_loss import _FedClientSelector_HighLoss
from selection.methods._fed_client_selector_random import _FedClientSelector_Random
from selection.fed_client_selector_abc import AbstractFedClientSelector

'''
' Fed client selector
'''


class FedClientSelector:

    @staticmethod
    def create(client_list: list, selection_method: EFedClientSelectMethod) -> AbstractFedClientSelector:        
        
        match selection_method:
            case EFedClientSelectMethod.all:
                selector = _FedClientSelector_All(client_list)
            case EFedClientSelectMethod.high_loss:
                selector = _FedClientSelector_HighLoss(client_list)
            case EFedClientSelectMethod.random:
                selector = _FedClientSelector_Random(client_list)                
            case _:
                raise Exception(f"Fed client selection type ({selection_method}) invalid")        
        return selector
