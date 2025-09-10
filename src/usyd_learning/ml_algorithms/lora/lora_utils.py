import torch.nn as nn
import torch
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
from .impl.lora_linear import LoRALinear

class LoRAUtils:
    @staticmethod
    def set_lora_mode_for_model(model: nn.Module, mode: str) -> None:
        """
        Set LoRA mode for all LoRALinear modules inside the model.

            Args:
                model: Target nn.Module that may contain LoRALinear submodules.
                mode:  Mode string understood by LoRALinear.set_lora_mode (e.g., "train", "freeze", "merge", etc.).
            """
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.set_lora_mode(mode)

    @staticmethod
    def get_lora_ranks(
        model: nn.Module,
        suffix_A: str = "lora_A",
        suffix_B: str = "lora_B",
    ) -> Dict[str, int]:
        """
        Scan model parameters and infer the LoRA rank per layer prefix.

        Conventions:
        - A-parameter is named "<prefix>.<suffix_A>" with shape [r, in].
        - B-parameter is named "<prefix>.<suffix_B>" with shape [out, r].
        Typically r = A.shape[0] = B.shape[1].

        Args:
            model:     Model that holds LoRA parameters.
            suffix_A:  Suffix for LoRA A matrix parameter name (default: "lora_A").
            suffix_B:  Suffix for LoRA B matrix parameter name (default: "lora_B").

        Returns:
            Mapping {layer_prefix: rank}. If only one side exists, its dimension is used.
            If both sides exist but ranks disagree, a ValueError is raised.
        """
        ranks: Dict[str, int] = {}
        lora_A_params: Dict[str, torch.Tensor] = {}
        lora_B_params: Dict[str, torch.Tensor] = {}

        # Collect A/B params keyed by prefix
        for name, param in model.named_parameters():
            if name.endswith(suffix_A):
                prefix = name[: -(len(suffix_A) + 1)]  # strip ".lora_A"
                lora_A_params[prefix] = param
            elif name.endswith(suffix_B):
                prefix = name[: -(len(suffix_B) + 1)]  # strip ".lora_B"
                lora_B_params[prefix] = param

        # Infer rank per prefix
        all_prefixes = set(lora_A_params.keys()) | set(lora_B_params.keys())
        for prefix in all_prefixes:
            r: Optional[int] = None
            if prefix in lora_A_params:
                r = int(lora_A_params[prefix].shape[0])  # A: [r, in]
            if prefix in lora_B_params:
                r_B = int(lora_B_params[prefix].shape[1])  # B: [out, r]
                r = r if r is not None else r_B
                # If both exist, they must agree (typical LoRA constraint)
                if r != r_B:
                    raise ValueError(
                        f"Inconsistent LoRA rank for '{prefix}': A={r}, B={r_B}"
                    )
            ranks[prefix] = int(r) if r is not None else 0

        return ranks

    @staticmethod
    def svd_split(
        weight: torch.Tensor,
        r: int,
        method: str = "sqrt",                 # "sqrt":  B = U * sqrt(S),  A = sqrt(S) * V^T
                                            # "full":  B = U * S,        A = V^T
        upcast_min_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Low-rank factorization by truncated SVD that returns LoRA-style (A, B).

        Accepts:
            - 2D weight: [out, in]
            - 4D conv weight: [out_c, in_c, kH, kW] (internally flattened to [out, in])

        Returns:
            (A, B) where:
            - A: [r, in]
            - B: [out, r]
            For conv weights, A/B are returned in 2D; callers should reshape ΔW back
            to [out_c, in_c, kH, kW] during forward reconstruction if needed.

        Notes:
            - Half/bfloat16 inputs are upcast to 'upcast_min_dtype' for numerical stability.
            - The effective rank rr is clamped to [1, min(out, in)].
        """
        if weight.dim() == 2:
            out_dim, in_dim = weight.shape
            W2d = weight
            reshape_back = None
        elif weight.dim() == 4:
            out_c, in_c, kh, kw = weight.shape
            W2d = weight.reshape(out_c, in_c * kh * kw)
            out_dim, in_dim = W2d.shape
            # A/B are 2D; users reconstruct ΔW and reshape to (out_c, in_c, kh, kw) externally.
            reshape_back = (out_c, in_c, kh, kw)
        else:
            raise ValueError(f"svd_split only supports 2D/4D tensors, got {weight.dim()}D")

        rr = max(1, min(int(r), out_dim, in_dim))

        # Upcast for stability if needed
        orig_dtype = W2d.dtype
        work = W2d if W2d.dtype not in (torch.float16, torch.bfloat16) else W2d.to(upcast_min_dtype)

        # U: [out, k], S: [k], Vh: [k, in]; k = min(out, in)
        U, S, Vh = torch.linalg.svd(work, full_matrices=False)
        U_r  = U[:, :rr]
        S_r  = S[:rr]
        Vh_r = Vh[:rr, :]

        if method == "sqrt":
            S_sqrt = torch.sqrt(torch.clamp(S_r, min=0))
            B = U_r * S_sqrt.unsqueeze(0)      # [out, r]
            A = S_sqrt.unsqueeze(1) * Vh_r     # [r, in]
        elif method == "full":
            B = U_r * S_r.unsqueeze(0)         # [out, r]
            A = Vh_r                           # [r, in]
        else:
            raise ValueError(f"Unknown method: {method}")

        # Cast back to original dtype if upcasted
        if B.dtype != orig_dtype:
            B = B.to(orig_dtype)
            A = A.to(orig_dtype)

        return A, B

    @staticmethod
    def svd_split_global_weight(
        global_weight: Dict[str, torch.Tensor],
        rank_dict: Dict[str, int],
        *,
        lora_suffix_A: str = "lora_A",
        lora_suffix_B: str = "lora_B",
        sp_suffix: str = "sp_aggregated",
        svd_method: str = "sqrt",
    ) -> "OrderedDict[str, torch.Tensor]":
        """
        Decompose aggregated weights (e.g., '<prefix>.sp_aggregated') into LoRA A/B
        with target ranks provided by rank_dict.

        Output order per layer:
            <prefix>.weight  -> (optional) <prefix>.bias -> <prefix>.lora_A -> <prefix>.lora_B

        Behavior:
            - Only keys ending with f".{sp_suffix}" are processed.
            - For each such key, look up target rank from rank_dict[prefix].
            - Perform SVD with effective rank eff_r = min(target_r, out, in).
            - If target_r > eff_r, right-/down-pad with zeros to match target_r.

        Args:
            global_weight: Mapping of parameter names to tensors, including '<prefix>.sp_aggregated'.
            rank_dict:     Mapping {prefix: target_rank}. Must contain all prefixes to be split.
            lora_suffix_A / lora_suffix_B: Output suffixes for A/B parameters.
            sp_suffix:     Suffix that marks aggregated base+delta (default: 'sp_aggregated').
            svd_method:    'sqrt' or 'full' (see svd_split).

        Returns:
            OrderedDict of tensors in a stable, layer-grouped order.
        """
        out: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        for k, W in global_weight.items():
            if not k.endswith(f".{sp_suffix}"):
                continue

            prefix = k[: -len(sp_suffix) - 1]  # strip ".sp_aggregated"

            # Target rank must be explicitly provided
            if prefix not in rank_dict:
                raise KeyError(f"rank_dict is missing the rank for layer '{prefix}'")
            target_r = int(rank_dict[prefix])

            # 1) Base weight: prefer existing '<prefix>.weight'; otherwise use sp_aggregated as weight proxy.
            w_key = f"{prefix}.weight"
            out[w_key] = global_weight.get(w_key, W)

            # 2) Bias (optional)
            b_key = f"{prefix}.bias"
            if b_key in global_weight:
                out[b_key] = global_weight[b_key]

            # 3) SVD at effective rank, then zero-pad to target rank if necessary.
            eff_r = max(1, min(target_r, W.shape[0], W.shape[1]))
            A_eff, B_eff = LoRAUtils.svd_split(W, eff_r, method=svd_method)

            if eff_r < target_r:
                A_pad = A_eff.new_zeros((target_r, A_eff.shape[1]))
                B_pad = B_eff.new_zeros((B_eff.shape[0], target_r))
                A_pad[:eff_r, :] = A_eff
                B_pad[:, :eff_r] = B_eff
                A_eff, B_eff = A_pad, B_pad

            out[f"{prefix}.{lora_suffix_A}"] = A_eff
            out[f"{prefix}.{lora_suffix_B}"] = B_eff

        return out


# import torch.nn as nn
# import torch
# from typing import List, Dict, Tuple, Optional
# from collections import OrderedDict
# from .impl.lora_linear import LoRALinear

# class LoRAUtils():
#     @staticmethod
#     def set_lora_mode_for_model(model: nn.Module, mode: str):
#         for module in model.modules():
#             if isinstance(module, LoRALinear):
#                 module.set_lora_mode(mode)

#     @staticmethod
#     def get_lora_ranks(
#         model: nn.Module,
#         suffix_A: str = "lora_A",
#         suffix_B: str = "lora_B"
#     ) -> Dict[str, int]:
#         """
#         遍历模型参数，提取每个层的 LoRA 秩。

#         Args:
#             model (nn.Module): 包含 LoRA 参数的模型
#             suffix_A (str): LoRA A 矩阵的后缀 (默认 "lora_A")
#             suffix_B (str): LoRA B 矩阵的后缀 (默认 "lora_B")

#         Returns:
#             dict: {layer_prefix: rank}
#         """
#         ranks: Dict[str, int] = {}
#         lora_A_params = {}
#         lora_B_params = {}

#         # 遍历所有参数
#         for name, param in model.named_parameters():
#             if name.endswith(suffix_A):
#                 prefix = name[: -(len(suffix_A) + 1)]  # 去掉 ".lora_A"
#                 lora_A_params[prefix] = param
#             elif name.endswith(suffix_B):
#                 prefix = name[: -(len(suffix_B) + 1)]  # 去掉 ".lora_B"
#                 lora_B_params[prefix] = param

#         # 计算 rank
#         all_prefixes = set(lora_A_params.keys()) | set(lora_B_params.keys())
#         for prefix in all_prefixes:
#             r = None
#             if prefix in lora_A_params:
#                 r = lora_A_params[prefix].shape[0]
#             if prefix in lora_B_params:
#                 r_B = lora_B_params[prefix].shape[1]
#                 r = r if r is not None else r_B
#                 # 双方都有的话取一致（一般 A.shape[0] == B.shape[1]）
#                 if r != r_B:
#                     raise ValueError(f"Inconsistent LoRA rank in {prefix}: "
#                                     f"A={r}, B={r_B}")
#             ranks[prefix] = int(r) if r is not None else 0

#         return ranks
    
#     @staticmethod
#     def svd_split(
#         weight: torch.Tensor,
#         r: int,
#         method: str = "sqrt",                 # "sqrt": B=U sqrt(S), A=sqrt(S) V^T；"full": B=U S, A=V^T
#         upcast_min_dtype: torch.dtype = torch.float32,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
        
#         if weight.dim() == 2:
#             out_dim, in_dim = weight.shape
#             W2d = weight
#             reshape_back = None
#         elif weight.dim() == 4:
#             out_c, in_c, kh, kw = weight.shape
#             W2d = weight.reshape(out_c, in_c * kh * kw)
#             out_dim, in_dim = W2d.shape
#             # conv 生成的 A/B 以二维形式返回；前向重建 ΔW 时按 (out,in,kh,kw) reshape 回去
#             reshape_back = (out_c, in_c, kh, kw)
#         else:
#             raise ValueError(f"svd_split only supports 2D/4D tensors, got {weight.dim()}D")

#         rr = max(1, min(int(r), out_dim, in_dim))

#         # 数值稳健：半精度先上采样
#         orig_dtype = W2d.dtype
#         work = W2d if W2d.dtype not in (torch.float16, torch.bfloat16) else W2d.to(upcast_min_dtype)

#         U, S, Vh = torch.linalg.svd(work, full_matrices=False)  # U(out,k), S(k,), Vh(k,in)
#         U_r  = U[:, :rr]
#         S_r  = S[:rr]
#         Vh_r = Vh[:rr, :]

#         if method == "sqrt":
#             S_sqrt = torch.sqrt(torch.clamp(S_r, min=0))
#             B = U_r * S_sqrt.unsqueeze(0)      # (out, r)
#             A = S_sqrt.unsqueeze(1) * Vh_r     # (r, in)
#         elif method == "full":
#             B = U_r * S_r.unsqueeze(0)         # (out, r)
#             A = Vh_r                            # (r, in)
#         else:
#             raise ValueError(f"Unknown method: {method}")

#         if B.dtype != orig_dtype:
#             B = B.to(orig_dtype)
#             A = A.to(orig_dtype)

#         # 注意：这里返回二维 A/B；对于 conv 的使用者，需在前向构造 ΔW 时再按 reshape_back 还原
#         return A, B

#     @staticmethod
#     def svd_split_global_weight(
#         global_weight: Dict[str, torch.Tensor],
#         rank_dict: Dict[str, int],
#         *,
#         lora_suffix_A: str = "lora_A",
#         lora_suffix_B: str = "lora_B",
#         sp_suffix: str = "sp_aggregated",
#         svd_method: str = "sqrt",
#     ) -> "OrderedDict[str, torch.Tensor]":
#         """
#         输出顺序：prefix.weight -> prefix.bias(若有) -> prefix.lora_A -> prefix.lora_B -> 下一层...
#         目标秩 r 必须来自 rank_dict；若 r > min(out,in)，将用 0 进行右/下方向的填充。
#         """
#         out = OrderedDict()

#         for k, W in global_weight.items():
#             if not k.endswith(f".{sp_suffix}"):
#                 continue

#             prefix = k[: -len(sp_suffix) - 1]  # 去掉 ".sp_aggregated"

#             # --- 目标秩：必须由 rank_dict 指定，不再用 min(W.shape) 兜底 ---
#             if prefix not in rank_dict:
#                 raise KeyError(f"rank_dict missing rank for layer '{prefix}'")
#             target_r = int(rank_dict[prefix])

#             # 1) weight：优先用已有的 prefix.weight；否则用 sp_aggregated 作为 weight
#             w_key = f"{prefix}.weight"
#             out[w_key] = global_weight.get(w_key, W)

#             # 2) bias（若有）
#             b_key = f"{prefix}.bias"
#             if b_key in global_weight:
#                 out[b_key] = global_weight[b_key]

#             # 3) 先按“有效秩”做 SVD，再对齐到“目标秩”
#             eff_r = max(1, min(target_r, W.shape[0], W.shape[1]))  # 有效秩 ≤ min(out,in)
#             A_eff, B_eff = LoRAUtils.svd_split(W, eff_r, method=svd_method)

#             # 若目标秩更大，则零填充到 (target_r, in) / (out, target_r)
#             if eff_r < target_r:
#                 A_pad = A_eff.new_zeros((target_r, A_eff.shape[1]))
#                 B_pad = B_eff.new_zeros((B_eff.shape[0], target_r))
#                 A_pad[:eff_r, :] = A_eff
#                 B_pad[:, :eff_r] = B_eff
#                 A_eff, B_eff = A_pad, B_pad

#             out[f"{prefix}.{lora_suffix_A}"] = A_eff
#             out[f"{prefix}.{lora_suffix_B}"] = B_eff

#         return out