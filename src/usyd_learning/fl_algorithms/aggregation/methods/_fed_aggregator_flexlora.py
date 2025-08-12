from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console

class FedAggregator_FlexLoRA(AbstractFedAggregator):

    #TODO: verify if this is the correct import path for the base class

    """
    Implements the FlexLoRA aggregation method.
    """

    def __init__(self, args: FedAggregatorArgs|None = None):
        super().__init__(args)
        self._aggregation_method = "flexrola"
        return
    
    def _before_aggregation(self) -> None:
        console.debug(f"[FlexLoRA] Starting aggregation with {len(self._aggregation_data_list)} clients...")

    def _do_aggregation(self) -> None:
        """
        Aggregate model weights using FlexLoRA.
        """
        self._aggregated_weight = {}
        
        for key in self._aggregation_data_list[0].keys():
            self._aggregated_weight[key] = sum(
                client['weight'] * self._aggregation_data_list[i][key] for i, client in enumerate(self._aggregation_data_list)
            ) / len(self._aggregation_data_list)
        return
    
    def _after_aggregation(self) -> None:
        console.debug(f"[FlexLoRA] Aggregation completed.")    