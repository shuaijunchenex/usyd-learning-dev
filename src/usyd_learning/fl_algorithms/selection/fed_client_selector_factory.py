from __future__ import annotations

from .methods._fed_client_selector_all import FedClientSelector_All
from .methods._fed_client_selector_high_loss import FedClientSelector_HighLoss
from .methods._fed_client_selector_random import FedClientSelector_Random
from .fed_client_selector_abc import AbstractFedClientSelector

'''
' Fed client selector factory
'''

class FedClientSelectorFactory:

    @staticmethod
    def create(config_dict: dict) -> AbstractFedClientSelector:
        
        if "client_selection" in config_dict:
            config = config_dict["client_selection"]
        else:
            config = config_dict

        selection_method = config.get("method", "random")
        random_seed = config.get("random_seed", 42)

        match selection_method:
            case "all":
                selector = FedClientSelector_All(random_seed)
            case "high_loss":
                selector = FedClientSelector_HighLoss(random_seed)
            case "random":
                selector = FedClientSelector_Random(random_seed)                
            case _:
                raise Exception(f"Fed client selection type ({selection_method}) invalid")        

        selector.select_number = config.get("number", 2)
        selector.select_round = config.get("round", 1)

        return selector
