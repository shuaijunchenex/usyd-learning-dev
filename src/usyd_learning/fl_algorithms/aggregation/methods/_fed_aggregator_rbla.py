import torch

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console

class FedAggregator_RBLA(AbstractFedAggregator):

    #TODO: verify if this is the correct import path for the base class

    """
    Implements the RBLA aggregation method.
    """
    
    def __init__(self, args: FedAggregatorArgs|None = None):
        super().__init__(args)
        self._aggregation_method = "rbla"
        self._lora_suffixes = {"lora_A", "lora_B"}
        self._pad_mode = 'nan'

        return

    def _before_aggregation(self) -> None:
        console.debug(f"[RBLA] Starting aggregation with {len(self._aggregation_data_dict['weights'])} clients...")

    def _do_aggregation(self) -> None:
        """
        Aggregate model weights using RBLA.
        """

        self.aggregate_state_dicts(self._aggregation_data_dict['state_dicts'],
                                   weights=self._aggregation_data_dict['weights'],
                                   lora_suffixes=self._lora_suffixes,
                                   pad_mode=self._pad_mode)
        
        return
    
    def _after_aggregation(self) -> None:
        console.debug(f"[RBLA] Aggregation completed.")        

    def set_lora_suffixes(self, lora_suffixes: set[str]) -> None:
        """
        Set the suffixes that indicate LoRA parameters.
        """
        self._lora_suffixes = lora_suffixes

    def set_pad_mode(self, pad_mode: str) -> None:
        """
        Set the padding mode for LoRA tensor aggregation.
        
        Args:
            pad_mode: 'nan' for NaN padding; 'zero' for zero padding.
        """
        assert pad_mode in {"nan", "zero"}, f"Unsupported pad_mode: {pad_mode}"
        self._pad_mode = pad_mode

    @staticmethod
    def get_suffix(key: str) -> str:
        """Return the suffix after the last dot in a key string."""
        return key.rsplit('.', 1)[-1]

    @staticmethod
    def pad_tensors_to_max_shape(tensors: list[torch.Tensor], pad_mode: str = 'nan') -> torch.Tensor:
        """
        Pad tensors to the same shape using either 'nan' or 'zero' padding.

        Args:
            tensors: List of 2D tensors with potentially different shapes.
            pad_mode: 'nan' for NaN padding; 'zero' for zero padding.

        Returns:
            A padded 3D tensor of shape (N, max_rows, max_cols).
        """
        assert pad_mode in {"nan", "zero"}, f"Unsupported pad_mode: {pad_mode}"
        max_rows = max(t.shape[0] for t in tensors)
        max_cols = max(t.shape[1] for t in tensors)
        device = tensors[0].device
        dtype = tensors[0].dtype

        padded = []
        for t in tensors:
            fill_val = float('nan') if pad_mode == 'nan' else 0.0
            padded_t = torch.full((max_rows, max_cols), fill_val, dtype=dtype, device=device)
            padded_t[:t.shape[0], :t.shape[1]] = t
            padded.append(padded_t)
        return torch.stack(padded, dim=0)

    @staticmethod
    def aggregate_lora_tensors(tensors: list[torch.Tensor], weights: list[float], pad_mode: str = 'nan') -> torch.Tensor:
        """
        Aggregate a list of LoRA matrices using weighted average with padding.

        Args:
            tensors: List of 2D LoRA matrices.
            weights: Weight for each matrix.
            pad_mode: 'nan' or 'zero' padding for alignment.

        Returns:
            A 2D tensor as the aggregated result.
        """
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=tensors[0].device).view(-1, 1, 1)
        padded = FedAggregator_RBLA.pad_tensors_to_max_shape(tensors, pad_mode=pad_mode)

        if pad_mode == 'nan':
            valid_mask = ~torch.isnan(padded)
            padded = torch.nan_to_num(padded, nan=0.0)
            weighted_sum = (padded * weights_tensor).sum(dim=0)
            weight_mask = valid_mask * weights_tensor
            total_weight = weight_mask.sum(dim=0)
            total_weight[total_weight == 0] = 1.0  # Avoid division by zero
            return weighted_sum / total_weight
        else:  # zero padding
            weighted_sum = (padded * weights_tensor).sum(dim=0)
            total_weight = sum(weights)
            return weighted_sum / total_weight

    @staticmethod
    def aggregate_state_dicts(
        state_dicts: list[dict],
        weights: list[float] = None,
        lora_suffixes: set[str] = {"lora_A", "lora_B"},
        pad_mode: str = 'nan') -> dict:
        """
        Aggregate multiple state_dicts, applying LoRA-aware weighted averaging.

        Args:
            state_dicts: List of state_dicts from clients.
            weights: Aggregation weights for each client.
            lora_suffixes: Suffixes indicating LoRA parameters.
            pad_mode: Padding strategy: 'nan' or 'zero'.

        Returns:
            Aggregated state_dict.
        """
        if weights is None:
            weights = [1.0] * len(state_dicts)

        keys = state_dicts[0].keys()
        aggregated = {}

        for key in keys:
            values = [sd[key] for sd in state_dicts]
            suffix = FedAggregator_RBLA.get_suffix(key)

            if suffix in lora_suffixes:
                # LoRA parameter: aggregate with padding
                aggregated[key] = FedAggregator_RBLA.aggregate_lora_tensors(values, weights, pad_mode=pad_mode)
            else:
                # Normal parameter: standard weighted average
                stacked = torch.stack(values, dim=0)  # (N, ...)
                weight_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device).view(
                    -1, *[1] * (stacked.dim() - 1))
                weighted_sum = (stacked * weight_tensor).sum(dim=0)
                total_weight = sum(weights)
                aggregated[key] = weighted_sum / total_weight

        return aggregated

    @staticmethod
    def broadcast_lora_state_dict(global_sd: dict, local_sd: dict, lora_suffixes={"lora_A", "lora_B"}) -> dict:
        """
        Distribute global aggregated state_dict to a client by slicing LoRA matrices.

        Args:
            global_sd: Aggregated global state_dict (full-rank).
            local_sd: Local client model state_dict (used to determine rank).
            lora_suffixes: Set of suffixes that indicate LoRA parameters.

        Returns:
            New local state_dict with sliced LoRA parameters and others replaced directly.
        """
        new_local_sd = {}

        for key in local_sd:
            global_tensor = global_sd[key]
            local_tensor = local_sd[key]
            suffix = FedAggregator_RBLA.get_suffix(key)

            if suffix not in lora_suffixes:
                # Non-LoRA parameter: directly replace
                new_local_sd[key] = global_tensor.clone()
            else:
                # LoRA parameter: slice to match local rank
                if suffix == "lora_A":
                    r_local = local_tensor.shape[0]
                    new_local_sd[key] = global_tensor[:r_local, :].clone()
                elif suffix == "lora_B":
                    r_local = local_tensor.shape[1]
                    new_local_sd[key] = global_tensor[:, :r_local].clone()
                else:
                    raise ValueError(f"Unrecognized LoRA suffix: {suffix}")

        return new_local_sd
    