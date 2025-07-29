import torch

from collections import OrderedDict
from fed_server_aggregation_abc import AbstractFedServerAggregator
from fed_server_aggregation_method import EFedServerAggregationMethod

from ....ml_utils import console


class Aggregator_FedAvg(AbstractFedServerAggregator):
    """
    Implements the FedAvg aggregation method using weighted average of client updates.
    """

    def __init__(self, aggregation_data_list: list, device: str = "cpu"):
        super().__init__(aggregation_data_list, EFedServerAggregationMethod.fedavg)
        self.device = torch.device(device)

    def _before_aggregation(self) -> None:
        console.out(f"[FedAvg] Starting aggregation with {len(self._aggregation_data_list)} clients...")

    def _do_aggregation(self) -> None:
        sample_state_dict = self._aggregation_data_list[0][0]  # get one sample model
        new_weights = OrderedDict()

        # Init empty tensor for aggregation
        for key in sample_state_dict.keys():
            new_weights[key] = torch.zeros_like(sample_state_dict[key], device=self.device)

        total_data_vol = sum(vol for _, vol in self._aggregation_data_list)

        # Debug info
        for i, (_, vol) in enumerate(self._aggregation_data_list):
            console.info(f"  Client {i}: {vol} samples ({vol / total_data_vol * 100:.1f}%)")

        for state_dict, vol in self._aggregation_data_list:
            weight = vol / total_data_vol
            for key in new_weights:
                new_weights[key] += state_dict[key].to(self.device) * weight

        self._aggregated_weight = new_weights

        # Debug: first param mean
        first_param_name = next(iter(new_weights.keys()))
        console.info(f"[FedAvg] Aggregated first param mean: {new_weights[first_param_name].mean():.6f}")

    def _after_aggregation(self) -> None:
        console.out(f"[FedAvg] Aggregation completed.")
