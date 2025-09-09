import torch
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs


class FedAggregator_SP(AbstractFedAggregator):
    """
    Sum-Product aggregator:
      - 对每个 LoRA 前缀 (prefix.lora_A / prefix.lora_B)，先按客户端计算 ΔW_i = B_i @ A_i；
      - 按客户端权重对 ΔW_i 加权汇总为 ΔW_agg；
      - 最终 W_out = (加权汇总的 base W) + ΔW_agg；
      - 输出字典只包含基础权重键（如 'layer'），不包含 lora_A / lora_B。
    """

    def __init__(self, args: Optional[FedAggregatorArgs] = None):
        super().__init__(args)
        self._aggregation_method = "sp"
        self._lora_suffix_A = "lora_A"
        self._lora_suffix_B = "lora_B"
        self._normalize_weights: bool = True

    # ---------- Public config ----------
    def set_lora_suffixes(self, suffix_A: str = "lora_A", suffix_B: str = "lora_B") -> None:
        self._lora_suffix_A = suffix_A
        self._lora_suffix_B = suffix_B

    def set_normalize_weights(self, normalize: bool = True) -> None:
        self._normalize_weights = normalize

    # ---------- FedAvg-style data building ----------
    def build_data_list(self, aggregation_data_dict: dict) -> None:
        """Accept internal list like: [(state_dict, data_volume), ...]"""
        self._aggregation_data_list = list(aggregation_data_dict.values())

    def build_data_dict(self, aggregation_data_dict) -> None:
        """Accept list[(state_dict, data_volume)] or dict{'state_dicts': [...], 'weights': [...]}"""
        self._aggregation_data_dict = aggregation_data_dict

    # ---------- Helpers ----------
    @staticmethod
    def _suffix_of(key: str) -> str:
        return key.rsplit(".", 1)[-1]

    @staticmethod
    def _prefix_of(key: str) -> str:
        return key.rsplit(".", 1)[0]

    @staticmethod
    def _weighted_sum(tensors: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
        stacked = torch.stack(tensors, dim=0)
        w = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device).view(len(weights), *([1] * (stacked.dim() - 1)))
        return (stacked * w).sum(dim=0)

    def _collect_inputs(self) -> Tuple[List[Dict[str, torch.Tensor]], List[float]]:
        if hasattr(self, "_aggregation_data_list") and self._aggregation_data_list:
            pairs = self._aggregation_data_list
            sds = [sd for sd, vol in pairs]
            ws  = [float(vol) for sd, vol in pairs]
        elif hasattr(self, "_aggregation_data_dict") and isinstance(self._aggregation_data_dict, list):
            pairs = self._aggregation_data_dict
            sds = [sd for sd, vol in pairs]
            ws  = [float(vol) for sd, vol in pairs]
        else:
            agg = self._aggregation_data_dict
            sds = agg["state_dicts"]
            ws  = agg.get("weights", [1.0] * len(sds))

        if self._normalize_weights:
            tw = float(sum(ws))
            ws = [w / tw for w in ws] if tw > 0 else [1.0 / len(ws)] * len(ws)
        return sds, ws

    # ---------- Aggregation core ----------
    def _before_aggregation(self) -> None:
        return

    def _do_aggregation(self) -> None:
        """
        产出仅包含基础权重键的 OrderedDict：
          - 对非 LoRA 键：直接按权重加权求和；
          - 对存在 LoRA 的层：最终写入 key=prefix 的 W = base_W_agg + ΔW_agg，
            其中 ΔW_agg 基于拥有该 LoRA 对的客户端子集做 B@A 的加权求和。
        """
        state_dicts, weights = self._collect_inputs()
        dev = self._device
        sds_on_device: List[Dict[str, torch.Tensor]] = [{k: v.to(dev) for k, v in sd.items()} for sd in state_dicts]

        sample_keys = list(state_dicts[0].keys())
        aggregated: Dict[str, torch.Tensor] = {}

        # 1) 先对所有基础权重键做加权和（非 lora_* 且是张量）
        for key in sample_keys:
            suf = self._suffix_of(key)
            if suf in {self._lora_suffix_A, self._lora_suffix_B}:
                continue
            values = [sd[key] for sd in sds_on_device if key in sd and torch.is_tensor(sd[key])]
            if values:
                aggregated[key] = self._weighted_sum(values, weights[:len(values)])

        # 2) 收集每个 LoRA 前缀下各客户端的 (A,B) 对，并对 ΔW 做加权和
        pair_groups: Dict[str, List[Tuple[int, torch.Tensor, torch.Tensor]]] = {}
        for ci, sd in enumerate(sds_on_device):
            per_client_pairs: Dict[str, Dict[str, str]] = {}
            for k in sd.keys():
                suf = self._suffix_of(k)
                if suf == self._lora_suffix_A or suf == self._lora_suffix_B:
                    prefix = self._prefix_of(k)
                    per_client_pairs.setdefault(prefix, {})
                    per_client_pairs[prefix][suf] = k
            for prefix, d in per_client_pairs.items():
                if self._lora_suffix_A in d and self._lora_suffix_B in d:
                    kA, kB = d[self._lora_suffix_A], d[self._lora_suffix_B]
                    pair_groups.setdefault(prefix, []).append((ci, sd[kA], sd[kB]))

        for prefix, entries in pair_groups.items():
            if not entries:
                continue
            full_list, local_w = [], []
            for ci, A, B in entries:
                # ΔW_i = B @ A（形状 out×in）
                full_list.append(B @ A)
                local_w.append(weights[ci])

            dW = self._weighted_sum(full_list, local_w)

            # 最终写入 W = base_W_agg + dW ；若无 base，则仅写 dW
            base_key = prefix
            agg_key = f"{prefix}.sp_aggregated"
            if base_key in aggregated:
                aggregated[agg_key] = aggregated[base_key] + dW
            else:
                aggregated[agg_key] = dW

        # 3) 构造有序结果：按 sample_keys 顺序输出存在的基础键；其余追加
        ordered = OrderedDict((k, aggregated[k]) for k in sample_keys if k in aggregated and self._suffix_of(k) not in {self._lora_suffix_A, self._lora_suffix_B})
        for k, v in aggregated.items():
            if k not in ordered:
                ordered[k] = v

        self._aggregated_weight = ordered

    def _after_aggregation(self) -> None:
        return
    
    