import torch.nn as nn
import torch
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
from .impl.lora_linear import LoRALinear

class LoRAUtils():
    @staticmethod
    def set_lora_mode_for_model(model: nn.Module, mode: str):
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.set_lora_mode(mode)

    @staticmethod
    def get_lora_ranks(
        model: nn.Module,
        suffix_A: str = "lora_A",
        suffix_B: str = "lora_B"
    ) -> Dict[str, int]:
        """
        遍历模型参数，提取每个层的 LoRA 秩。

        Args:
            model (nn.Module): 包含 LoRA 参数的模型
            suffix_A (str): LoRA A 矩阵的后缀 (默认 "lora_A")
            suffix_B (str): LoRA B 矩阵的后缀 (默认 "lora_B")

        Returns:
            dict: {layer_prefix: rank}
        """
        ranks: Dict[str, int] = {}
        lora_A_params = {}
        lora_B_params = {}

        # 遍历所有参数
        for name, param in model.named_parameters():
            if name.endswith(suffix_A):
                prefix = name[: -(len(suffix_A) + 1)]  # 去掉 ".lora_A"
                lora_A_params[prefix] = param
            elif name.endswith(suffix_B):
                prefix = name[: -(len(suffix_B) + 1)]  # 去掉 ".lora_B"
                lora_B_params[prefix] = param

        # 计算 rank
        all_prefixes = set(lora_A_params.keys()) | set(lora_B_params.keys())
        for prefix in all_prefixes:
            r = None
            if prefix in lora_A_params:
                r = lora_A_params[prefix].shape[0]
            if prefix in lora_B_params:
                r_B = lora_B_params[prefix].shape[1]
                r = r if r is not None else r_B
                # 双方都有的话取一致（一般 A.shape[0] == B.shape[1]）
                if r != r_B:
                    raise ValueError(f"Inconsistent LoRA rank in {prefix}: "
                                    f"A={r}, B={r_B}")
            ranks[prefix] = int(r) if r is not None else 0

        return ranks
    
    @staticmethod
    def svd_split(
        weight: torch.Tensor,
        r: int,
        method: str = "sqrt",                 # "sqrt": B=U sqrt(S), A=sqrt(S) V^T；"full": B=U S, A=V^T
        upcast_min_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if weight.dim() == 2:
            out_dim, in_dim = weight.shape
            W2d = weight
            reshape_back = None
        elif weight.dim() == 4:
            out_c, in_c, kh, kw = weight.shape
            W2d = weight.reshape(out_c, in_c * kh * kw)
            out_dim, in_dim = W2d.shape
            # conv 生成的 A/B 以二维形式返回；前向重建 ΔW 时按 (out,in,kh,kw) reshape 回去
            reshape_back = (out_c, in_c, kh, kw)
        else:
            raise ValueError(f"svd_split only supports 2D/4D tensors, got {weight.dim()}D")

        rr = max(1, min(int(r), out_dim, in_dim))

        # 数值稳健：半精度先上采样
        orig_dtype = W2d.dtype
        work = W2d if W2d.dtype not in (torch.float16, torch.bfloat16) else W2d.to(upcast_min_dtype)

        U, S, Vh = torch.linalg.svd(work, full_matrices=False)  # U(out,k), S(k,), Vh(k,in)
        U_r  = U[:, :rr]
        S_r  = S[:rr]
        Vh_r = Vh[:rr, :]

        if method == "sqrt":
            S_sqrt = torch.sqrt(torch.clamp(S_r, min=0))
            B = U_r * S_sqrt.unsqueeze(0)      # (out, r)
            A = S_sqrt.unsqueeze(1) * Vh_r     # (r, in)
        elif method == "full":
            B = U_r * S_r.unsqueeze(0)         # (out, r)
            A = Vh_r                            # (r, in)
        else:
            raise ValueError(f"Unknown method: {method}")

        if B.dtype != orig_dtype:
            B = B.to(orig_dtype)
            A = A.to(orig_dtype)

        # 注意：这里返回二维 A/B；对于 conv 的使用者，需在前向构造 ΔW 时再按 reshape_back 还原
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
        输出顺序：prefix.weight -> prefix.bias(若有) -> prefix.lora_A -> prefix.lora_B -> 下一层...
        目标秩 r 必须来自 rank_dict；若 r > min(out,in)，将用 0 进行右/下方向的填充。
        """
        out = OrderedDict()

        for k, W in global_weight.items():
            if not k.endswith(f".{sp_suffix}"):
                continue

            prefix = k[: -len(sp_suffix) - 1]  # 去掉 ".sp_aggregated"

            # --- 目标秩：必须由 rank_dict 指定，不再用 min(W.shape) 兜底 ---
            if prefix not in rank_dict:
                raise KeyError(f"rank_dict missing rank for layer '{prefix}'")
            target_r = int(rank_dict[prefix])

            # 1) weight：优先用已有的 prefix.weight；否则用 sp_aggregated 作为 weight
            w_key = f"{prefix}.weight"
            out[w_key] = global_weight.get(w_key, W)

            # 2) bias（若有）
            b_key = f"{prefix}.bias"
            if b_key in global_weight:
                out[b_key] = global_weight[b_key]

            # 3) 先按“有效秩”做 SVD，再对齐到“目标秩”
            eff_r = max(1, min(target_r, W.shape[0], W.shape[1]))  # 有效秩 ≤ min(out,in)
            A_eff, B_eff = LoRAUtils.svd_split(W, eff_r, method=svd_method)

            # 若目标秩更大，则零填充到 (target_r, in) / (out, target_r)
            if eff_r < target_r:
                A_pad = A_eff.new_zeros((target_r, A_eff.shape[1]))
                B_pad = B_eff.new_zeros((B_eff.shape[0], target_r))
                A_pad[:eff_r, :] = A_eff
                B_pad[:, :eff_r] = B_eff
                A_eff, B_eff = A_pad, B_pad

            out[f"{prefix}.{lora_suffix_A}"] = A_eff
            out[f"{prefix}.{lora_suffix_B}"] = B_eff

        return out