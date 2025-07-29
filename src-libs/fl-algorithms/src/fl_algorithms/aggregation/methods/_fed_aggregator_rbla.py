import torch
from collections import OrderedDict
from fed_server_aggregation_abc import AbstractFedServerAggregator
from fed_server_aggregation_method import EFedServerAggregationMethod
from typing import List, Dict, Optional

class _Aggregator_RBLA(AbstractFedServerAggregator):
    """
    Implements the RBLA aggregation method using rank-based structured LoRA parameter aggregation.
    """

    def __init__(self, aggregation_data_list: list, device: str = "cpu"):
        super().__init__(aggregation_data_list, EFedServerAggregationMethod.rbla)
        self.device = torch.device(device)

    def _before_aggregation(self):
        print(f"[RBLA] Starting aggregation with {len(self._aggregation_data_list)} clients...")

    def _do_aggregation(self):
        """
        Perform the RBLA aggregation using rank-based structured LoRA aggregation.
        Aggregates based on data volume as client weight.
        """
        client_model_data_list = [client_data[0] for client_data in self._aggregation_data_list]
        client_data_volume_list = [client_data[1] for client_data in self._aggregation_data_list]

        total_volume = sum(client_data_volume_list)
        if total_volume == 0:
            print("[RBLA] Warning: Total data volume is zero, using uniform weights.")
            client_weight_list = [1.0 / len(client_data_volume_list)] * len(client_data_volume_list)
        else:
            client_weight_list = [v / total_volume for v in client_data_volume_list]

        self._aggregated_result = self.rank_based_structured_lora_aggregation(
            client_model_data_list, client_weight_list
        )


    def _after_aggregation(self):
        print(f"[RBLA] Aggregation completed.")

    def pad_to_match_shape(self, weight_tensor_list: List[torch.Tensor], padding_mode: str = "none padding") -> List[torch.Tensor]:
        max_shape = [0] * len(weight_tensor_list[0].shape)
        for weight_tensor in weight_tensor_list:
            current_shape = list(weight_tensor.shape)
            max_shape = [max(max_dim, curr_dim) for max_dim, curr_dim in zip(max_shape, current_shape)]

        padded_tensor_list = []
        for weight_tensor in weight_tensor_list:
            pad_value = float("nan") if padding_mode == "none padding" else 0.0
            padded_tensor = torch.full(max_shape, pad_value, dtype=weight_tensor.dtype, device=weight_tensor.device)
            slices = tuple(slice(0, s) for s in weight_tensor.shape)
            padded_tensor[slices] = weight_tensor
            padded_tensor_list.append(padded_tensor)

        return padded_tensor_list

    def fast_layerwise_weighted_sparse_matrix_aggregation(self, weight_tensor_list: List[torch.Tensor], weight_list: List[float], layer_type: str) -> Optional[torch.Tensor]:
        if layer_type in ['batchnorm', 'lora']:
            padded_tensor_list = self.pad_to_match_shape(weight_tensor_list)
            stacked_weights = torch.stack(padded_tensor_list, dim=0).double()
            weight_tensor = torch.tensor(weight_list, dtype=stacked_weights.dtype, device=stacked_weights.device)
            weight_tensor = weight_tensor.view(-1, *([1] * (stacked_weights.dim() - 1)))
            valid_mask = (~torch.isnan(stacked_weights)).double()
            weighted_sum = torch.nansum(stacked_weights * weight_tensor, dim=0)
            total_weight = torch.nansum(valid_mask * weight_tensor, dim=0)
            return torch.where(total_weight != 0, weighted_sum / total_weight, torch.tensor(float("nan"), dtype=stacked_weights.dtype, device=stacked_weights.device))

        elif layer_type == "conv2d":
            valid_weights = [w for w in weight_tensor_list if w is not None]
            if not valid_weights:
                return None
            stacked_weights = torch.stack(valid_weights, dim=0).double()
            weight_tensor = torch.tensor(weight_list[:len(valid_weights)], dtype=stacked_weights.dtype, device=stacked_weights.device)
            weight_tensor = weight_tensor.view(-1, *[1] * (stacked_weights.dim() - 1))
            weighted_sum = torch.sum(stacked_weights * weight_tensor, dim=0)
            total_weight = torch.sum(weight_tensor, dim=0)
            return weighted_sum / total_weight

        return None

    def _determine_layer_type(self, layer_name: str, param_dict: Dict[str, torch.Tensor]) -> str:
        if 'layer_type' in param_dict:
            layer_type = param_dict['layer_type']
            if isinstance(layer_type, str):
                if layer_type in ["batchnorm1d", "batchnorm2d"]:
                    return "batchnorm"
                elif "lora" in layer_type.lower():
                    return "lora"
                elif "conv2d" in layer_type.lower():
                    return "conv2d"
        if "lora_a" in param_dict or "lora_A" in param_dict:
            return "lora"
        elif "bn" in layer_name.lower() or "batchnorm" in layer_name.lower():
            return "batchnorm"
        elif "conv" in layer_name.lower():
            return "conv2d"
        return "default"

    def rank_based_structured_lora_aggregation(self, client_model_data_list: List[Dict[str, Dict[str, torch.Tensor]]], client_weight_list: List[float]) -> Dict[str, Dict[str, torch.Tensor]]:
        if not client_model_data_list:
            return {}

        all_layer_names = set()
        for client_data in client_model_data_list:
            all_layer_names.update(client_data.keys())

        aggregated_model = {}
        for layer_name in all_layer_names:
            present_clients = [layer_name in client for client in client_model_data_list]
            if not any(present_clients):
                continue

            all_param_keys = set()
            for i, present in enumerate(present_clients):
                if present:
                    all_param_keys.update(client_model_data_list[i][layer_name].keys())

            param_keys_to_aggregate = set()
            for i, present in enumerate(present_clients):
                if present:
                    for key in all_param_keys:
                        if isinstance(client_model_data_list[i][layer_name].get(key), torch.Tensor):
                            param_keys_to_aggregate.add(key)

            rep_idx = present_clients.index(True)
            rep_params = client_model_data_list[rep_idx][layer_name]
            layer_type = self._determine_layer_type(layer_name, rep_params)

            aggregated_layer = {}
            for param_key in param_keys_to_aggregate:
                if param_key == "layer_type":
                    continue
                active_weights, active_tensors = [], []
                for i, present in enumerate(present_clients):
                    if present and param_key in client_model_data_list[i][layer_name]:
                        active_weights.append(client_weight_list[i])
                        active_tensors.append(client_model_data_list[i][layer_name][param_key])
                if not active_tensors:
                    continue
                aggregated_tensor = self.fast_layerwise_weighted_sparse_matrix_aggregation(active_tensors, active_weights, layer_type)
                if aggregated_tensor is not None:
                    aggregated_layer[param_key] = aggregated_tensor

            if 'layer_type' in rep_params:
                aggregated_layer['layer_type'] = rep_params['layer_type']

            aggregated_model[layer_name] = aggregated_layer

        return aggregated_model

    def apply_aggregated_params(self, model, aggregated_params: Dict[str, Dict[str, torch.Tensor]]) -> None:
        for layer_name, params in aggregated_params.items():
            module = dict(model.named_modules()).get(layer_name, None)
            if module is None:
                continue
            for param_name, param_value in params.items():
                if param_name == "layer_type":
                    continue
                if hasattr(module, param_name):
                    target = getattr(module, param_name)
                    if isinstance(target, torch.nn.Parameter):
                        with torch.no_grad():
                            if target.shape == param_value.shape:
                                target.copy_(param_value)
                            else:
                                print(f"[Warning] Shape mismatch in {layer_name}.{param_name}: expected {target.shape}, got {param_value.shape}")
                    else:
                        setattr(module, param_name, param_value)