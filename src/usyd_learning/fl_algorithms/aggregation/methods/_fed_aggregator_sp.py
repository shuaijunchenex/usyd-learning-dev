import torch
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional

from ..fed_aggregator_abc import AbstractFedAggregator
from ..fed_aggregator_args import FedAggregatorArgs
from ....ml_utils import console


class FedAggregator_SP(AbstractFedAggregator):
    """
    Sum-Product LoRA Aggregator (SP):
    - For each LoRA pair (prefix.lora_A, prefix.lora_B), compose full delta: ΔW_i
      by multiplying the client's LoRA matrices (order is auto-detected).
    - Aggregate ΔW across clients with normalized weights.
    - Factorize aggregated ΔW back to (B_agg, A_agg) via SVD, so that the
      returned state_dict keeps the same keys (...lora_A / ...lora_B).
    - Non-LoRA params are aggregated by weighted average (FedAvg-style).

    API-compatible with FedAggregator_FedAvg / FedAggregator_RBLA:
      - build_data_list / build_data_dict
      - _do_aggregation() sets self._aggregated_weight (OrderedDict)
    """

    def __init__(self, args: Optional[FedAggregatorArgs] = None):
        super().__init__(args)
        self._aggregation_method = "sp"
        self._lora_suffix_A = "lora_A"
        self._lora_suffix_B = "lora_B"
        self._normalize_weights: bool = True   # normalize weights to sum=1
        self._rank_mode: str = "max_rank"      # {"max_rank", "fixed"}
        self._fixed_rank: Optional[int] = None
        return

    # ---------- Public config ----------
    def set_lora_suffixes(self, suffix_A: str = "lora_A", suffix_B: str = "lora_B") -> None:
        self._lora_suffix_A = suffix_A
        self._lora_suffix_B = suffix_B

    def set_rank_mode(self, mode: str = "max_rank", fixed_rank: Optional[int] = None) -> None:
        """
        mode:
          - "max_rank": use the maximum client LoRA rank for this pair
          - "fixed": use 'fixed_rank' (will be clamped to [1, min(out_dim, in_dim)])
        """
        assert mode in {"max_rank", "fixed"}, f"Unsupported rank_mode: {mode}"
        self._rank_mode = mode
        self._fixed_rank = fixed_rank

    def set_normalize_weights(self, normalize: bool = True) -> None:
        self._normalize_weights = normalize

    # ---------- FedAvg-style data building ----------
    def build_data_list(self, aggregation_data_dict: dict) -> None:
        """
        Accept internal list like: [(state_dict, data_volume), ...]
        """
        self._aggregation_data_list = list(aggregation_data_dict.values())
        return

    def build_data_dict(self, aggregation_data_dict) -> None:
        """
        Can be:
          - list[(state_dict, data_volume)]
          - dict {'state_dicts': [...], 'weights': [...]}
        """
        self._aggregation_data_dict = aggregation_data_dict

    # ---------- Helpers ----------
    @staticmethod
    def _suffix_of(key: str) -> str:
        return key.rsplit(".", 1)[-1]

    @staticmethod
    def _prefix_of(key: str) -> str:
        return key.rsplit(".", 1)[0]

    @staticmethod
    def _compose_lora_full(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compose to full ΔW. We auto-detect the correct multiplication order.
        Typical LoRA naming: A: (r, in), B: (out, r) => ΔW = B @ A
        We also accept reversed naming by checking shapes.
        """
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError(f"LoRA A/B must be 2D, got A={A.dim()}D, B={B.dim()}D")

        if B.shape[1] == A.shape[0]:
            # standard: B(out, r) @ A(r, in) -> (out, in)
            return B @ A
        elif A.shape[1] == B.shape[0]:
            # reversed naming: A(out, r) @ B(r, in) -> (out, in)
            return A @ B
        else:
            raise ValueError(
                f"Incompatible LoRA shapes for matmul: A{tuple(A.shape)}, B{tuple(B.shape)} "
                f"(expected B.shape[1]==A.shape[0] or A.shape[1]==B.shape[0])"
            )

    @staticmethod
    def _svd_factorize(full: torch.Tensor, target_rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Factorize full (out, in) into (B, A) with B(out, r) and A(r, in) such that B @ A ≈ full.
        Use SVD: full = U S V^T. Let B = U S^{1/2}, A = S^{1/2} V^T (truncated to r).
        """
        if full.dim() != 2:
            raise ValueError(f"SVD factorization expects 2D matrix, got {full.dim()}D")
        out_dim, in_dim = full.shape
        r = max(1, min(target_rank, out_dim, in_dim))

        # Torch SVD
        U, S, Vh = torch.linalg.svd(full, full_matrices=False)
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]

        S_sqrt = torch.sqrt(S_r)
        # B: (out, r) = U * sqrt(S)
        B = U_r * S_sqrt.unsqueeze(0)
        # A: (r, in)  = sqrt(S) * V^T
        A = (S_sqrt.unsqueeze(1) * Vh_r)
        return B, A

    @staticmethod
    def _reconstruct_lora_matrix(full_rank_weight, rank) -> torch.Tensor:
        return

    @staticmethod
    def _weighted_avg_tensors(tensors: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
        stacked = torch.stack(tensors, dim=0)
        w = torch.as_tensor(weights, dtype=stacked.dtype, device=stacked.device).view(len(weights), *([1] * (stacked.dim() - 1)))
        return (stacked * w).sum(dim=0)

    # ---------- Aggregation core ----------
    def _before_aggregation(self) -> None:
        # console.debug(f"[SP] Starting aggregation ...")
        return

    def _collect_inputs(self) -> Tuple[List[Dict[str, torch.Tensor]], List[float]]:
        state_dicts, weights = None, None

        # Preferred internal list
        if hasattr(self, "_aggregation_data_list") and self._aggregation_data_list:
            pairs = self._aggregation_data_list
            state_dicts = [sd for sd, vol in pairs]
            weights    = [float(vol) for sd, vol in pairs]

        # List of pairs in dict slot
        elif hasattr(self, "_aggregation_data_dict") and isinstance(self._aggregation_data_dict, list):
            pairs = self._aggregation_data_dict
            state_dicts = [sd for sd, vol in pairs]
            weights    = [float(vol) for sd, vol in pairs]

        # Legacy dict form
        elif hasattr(self, "_aggregation_data_dict") and isinstance(self._aggregation_data_dict, dict):
            agg = self._aggregation_data_dict
            state_dicts = agg["state_dicts"]
            weights     = agg.get("weights", None)
            if weights is None:
                weights = [1.0] * len(state_dicts)
        else:
            raise ValueError("[SP] No aggregation data found. Provide a list of (state_dict, data_volume) "
                             "or dict {'state_dicts': [...], 'weights': [...]}.")

        if weights is None:
            weights = [1.0] * len(state_dicts)

        if self._normalize_weights:
            tw = float(sum(weights))
            weights = [w / tw for w in weights] if tw > 0 else [1.0 / len(weights)] * len(weights)

        return state_dicts, weights

    def _do_aggregation(self) -> None:
        """
        Sum-Product aggregation:
          - Identify LoRA pairs by suffixes (.lora_A / .lora_B) sharing the same prefix.
          - For each pair, compose full ΔW_i for each client, do weighted sum => ΔW_agg.
          - Factorize ΔW_agg back to (B_agg, A_agg) with target rank.
          - Non-LoRA params: weighted average (FedAvg).
        """
        state_dicts, weights = self._collect_inputs()
        dev = self._device
        sds_on_device: List[Dict[str, torch.Tensor]] = [
            {k: v.to(dev) for k, v in sd.items()} for sd in state_dicts
        ]

        sample_keys = list(state_dicts[0].keys())
        aggregated: Dict[str, torch.Tensor] = {}

        # 1) 先按前缀把 LoRA 成对分组
        #    prefix -> list of per-client (A_tensor, B_tensor, A_key, B_key)
        pair_groups: Dict[str, List[Tuple[torch.Tensor, torch.Tensor, str, str]]] = {}

        for ci, sd in enumerate(sds_on_device):
            # 建立 prefix -> {A_key,B_key}
            per_client_pairs: Dict[str, Dict[str, str]] = {}
            for k in sd.keys():
                suf = self._suffix_of(k)
                if suf == self._lora_suffix_A or suf == self._lora_suffix_B:
                    prefix = self._prefix_of(k)
                    per_client_pairs.setdefault(prefix, {})
                    per_client_pairs[prefix][suf] = k

            # 收集成对的键
            for prefix, d in per_client_pairs.items():
                if self._lora_suffix_A in d and self._lora_suffix_B in d:
                    kA = d[self._lora_suffix_A]
                    kB = d[self._lora_suffix_B]
                    A = sd[kA]
                    B = sd[kB]
                    pair_groups.setdefault(prefix, []).append((A, B, kA, kB))
                else:
                    console.warn(f"[SP] Client {ci} has incomplete LoRA pair for '{prefix}', skipped in SP aggregation.")

        # 2) 对每个前缀完成 ΔW 聚合，再 SVD 拆回 A,B（并回写到 aggregated 中）
        for prefix, tuples in pair_groups.items():
            if len(tuples) == 0:
                continue

            # 2.1 逐 client 组成 full ΔW_i
            full_list: List[torch.Tensor] = []
            lora_ranks: List[int] = []
            for i, (A, B, _, _) in enumerate(tuples):
                full = self._compose_lora_full(A, B)  # (out, in)
                full_list.append(full)
                # 记录该 client 的秩候选（取 A/B 的 r 维）
                r_candidates = []
                if self._suffix_of(tuples[i][2]) == self._lora_suffix_A:  # A key
                    r_candidates.append(A.shape[0])
                if self._suffix_of(tuples[i][3]) == self._lora_suffix_B:  # B key
                    r_candidates.append(B.shape[1])
                if r_candidates:
                    lora_ranks.append(min(r_candidates))

            # 2.2 加权和（或加权平均，取决于 _normalize_weights）
            #     这里我们使用“权重已归一化”的和 => 实际上是加权平均
            W = self._weighted_avg_tensors(full_list, weights[:len(full_list)])

            # 2.3 选择目标 rank
            out_dim, in_dim = W.shape
            if self._rank_mode == "fixed" and self._fixed_rank is not None:
                target_r = self._fixed_rank
            else:
                # 默认：用客户端最大秩，但不超过矩阵本身的可达秩
                target_r = max(lora_ranks) if lora_ranks else min(out_dim, in_dim)

            target_r = max(1, min(target_r, out_dim, in_dim))

            # 2.4 SVD 拆回 B(out,r), A(r,in)
            B_agg, A_agg = self._svd_factorize(W, target_r)

            # 2.5 放回 aggregated（用该前缀的标准 key 命名）
            kA = f"{prefix}.{self._lora_suffix_A}"
            kB = f"{prefix}.{self._lora_suffix_B}"
            aggregated[kA] = A_agg
            aggregated[kB] = B_agg

            console.debug(f"[SP] {prefix} -> ΔW rank≈{torch.linalg.matrix_rank(W).item()} | used r={target_r}")

        # 3) 处理非 LoRA 参数：按 FedAvg 方式加权平均
        for key in sample_keys:
            suf = self._suffix_of(key)
            if suf in {self._lora_suffix_A, self._lora_suffix_B}:
                # 已由 SVD 结果覆盖
                continue

            values = [sd[key] for sd in sds_on_device]
            aggregated[key] = self._weighted_avg_tensors(values, weights)

        # 4) 用 sample key 顺序构造 OrderedDict，保证可直接 load_state_dict
        ordered = OrderedDict((k, aggregated[k]) for k in sample_keys if k in aggregated)
        # 如果某些 LoRA 键在首个样本中不存在（极罕见），也附加进去
        for k, v in aggregated.items():
            if k not in ordered:
                ordered[k] = v

        self._aggregated_weight = ordered

        # 打个小日志
        first_param_name = next(iter(ordered.keys()))
        try:
            console.debug(f"[SP] Aggregated first param mean: {ordered[first_param_name].mean():.6f}")
        except Exception:
            pass

    def _after_aggregation(self) -> None:
        console.debug("[SP] Aggregation completed.")
