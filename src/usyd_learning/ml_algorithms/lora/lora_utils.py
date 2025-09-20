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

    @staticmethod
    def convert_lora_for_sp_inference(
        base_state_dict: dict,
        lora_template_state_dict: dict,
        suffix_a: str = "lora_A",
        suffix_b: str = "lora_B",
        remove_key: str = "sp_aggregated",
        clone_base: bool = True,
        overwrite_existing: bool = True,
    ) -> OrderedDict:
        """
        Convert a base state_dict for SP-style inference:
        1) Replace each `{prefix}.weight` with `{prefix}.sp_aggregated` when available.
        2) Remove all keys whose last dotted component is `remove_key` (default: 'sp_aggregated').
        3) Ensure LoRA A/B keys (matching the template) exist in the output, initialized to zeros.
           - A/B zeros are dtype/device-aligned to `{prefix}.weight` if it exists; otherwise aligned to template.

        Args:
            base_state_dict: The model state_dict containing base weights and possibly `{prefix}.sp_aggregated`.
            lora_template_state_dict: A state_dict that indicates which LoRA A/B keys (shapes) should exist.
            suffix_a / suffix_b: LoRA suffixes to detect (e.g., 'lora_A'/'lora_B' or 'lora_down'/'lora_up').
            remove_key: Keys whose last dotted component equals this will be removed (default: 'sp_aggregated').
            clone_base: If True, tensors are cloned/detached; otherwise references are kept.
            overwrite_existing: If True, existing A/B in base will be overwritten with zeros.

        Returns:
            OrderedDict: The converted state_dict ready for SP inference.
        """

        # ---------- Helpers ----------
        def last_component(k: str) -> str:
            """Return the last dotted component of a key."""
            return k.rsplit(".", 1)[-1]

        def key_prefix(k: str) -> str:
            """Return the prefix before the last dot; if no dot, return ''."""
            return k.rsplit(".", 1)[0] if "." in k else ""

        def is_exact_remove_key(k: str) -> bool:
            """Remove only if the *last* component equals `remove_key`."""
            return last_component(k) == remove_key

        def is_lora_key(k: str) -> bool:
            """Check if key ends with LoRA A/B suffix."""
            return k.endswith(suffix_a) or k.endswith(suffix_b)

        def split_lora_prefix(k: str) -> Tuple[Optional[str], Optional[str]]:
            """Return (prefix, suffix) if `k` is a LoRA A/B key, else (None, None)."""
            if k.endswith(suffix_a):
                return k[: -len(suffix_a)].rstrip("."), suffix_a
            if k.endswith(suffix_b):
                return k[: -len(suffix_b)].rstrip("."), suffix_b
            return None, None

        # ---------- Stage 0: Index `{prefix}.sp_aggregated` before we drop them ----------
        # We only use entries whose LAST component equals `remove_key` to map prefixes.
        sp_map = {}
        for k, v in base_state_dict.items():
            if is_exact_remove_key(k) and torch.is_tensor(v):
                pref = key_prefix(k)  # '{prefix}.sp_aggregated' -> '{prefix}'
                sp_map[pref] = v

        # ---------- Stage 1: Copy base and simultaneously DROP all '*.sp_aggregated' ----------
        # We drop any key whose last dotted component == remove_key.
        new_sd = OrderedDict()
        for k, v in base_state_dict.items():
            if is_exact_remove_key(k):
                continue  # strip SP cache
            if clone_base and torch.is_tensor(v):
                new_sd[k] = v.detach().clone()
            else:
                new_sd[k] = v

        # ---------- Stage 2: Overwrite `{prefix}.weight` with `{prefix}.sp_aggregated` when present ----------
        # For each prefix found in sp_map, replace the base weight if possible.
        for pref, sp_tensor in sp_map.items():
            weight_key = f"{pref}.weight"
            if weight_key in new_sd and torch.is_tensor(new_sd[weight_key]):
                # Align sp tensor to existing weight dtype/device (safer if base dtype differs from template)
                tgt = new_sd[weight_key]
                sp_aligned = sp_tensor.to(dtype=tgt.dtype, device=tgt.device)
                new_sd[weight_key] = sp_aligned.detach().clone() if clone_base else sp_aligned
            else:
                # If there is no base weight, still install it (use sp dtype/device as-is).
                new_sd[weight_key] = sp_tensor.detach().clone() if clone_base else sp_tensor

        # ---------- Stage 3: Ensure LoRA A/B keys exist (zeros), shapes from template ----------
        added = 0
        for k_tmpl, v_tmpl in lora_template_state_dict.items():
            if not is_lora_key(k_tmpl):
                continue

            pref, sfx = split_lora_prefix(k_tmpl)
            if pref is None:
                continue

            # Prefer aligning zeros to the now-final `{prefix}.weight` if present,
            # otherwise align to the template tensor dtype/device.
            ref = new_sd.get(f"{pref}.weight", v_tmpl)
            if not torch.is_tensor(ref):
                ref = v_tmpl

            zero_like = torch.zeros_like(v_tmpl, dtype=ref.dtype, device=ref.device)
            if overwrite_existing or (k_tmpl not in new_sd):
                new_sd[k_tmpl] = zero_like
                added += 1

        if added == 0:
            # If the template has no LoRA keys, inform the caller (same behavior as before).
            raise ValueError(
                f"No LoRA parameters ending with '{suffix_a}' or '{suffix_b}' were found in the template. "
                f"(All '{remove_key}' caches have been removed, and weights were updated from them where present.)"
            )

        return new_sd

    # @staticmethod
    # def convert_lora_for_sp_inference(
    #     base_state_dict: dict,
    #     lora_template_state_dict: dict,
    #     suffix_a: str = "lora_A",
    #     suffix_b: str = "lora_B",
    #     remove_key: str = "sp_aggregated",
    #     clone_base: bool = True,
    #     overwrite_existing: bool = True,
    # ) -> OrderedDict:
    #     """
    #     在 base_state_dict 中补齐与模板相同形状的 LoRA A/B（置零），并移除所有 *.sp_aggregated。
    #     - suffix_a/suffix_b: LoRA 后缀（如 'lora_A','lora_B' 或 'lora_down','lora_up'）
    #     - remove_key: 需要移除的键名（末段），默认 'sp_aggregated'
    #     - overwrite_existing: 如 base 已有同名 A/B，是否用全 0 覆盖
    #     """
    #     def has_remove_key(k: str) -> bool:
    #         # 末段或整键包含都移除更稳妥
    #         return (k.endswith(remove_key)) or (remove_key in k.split("."))

    #     def is_lora_key(k: str) -> bool:
    #         return k.endswith(suffix_a) or k.endswith(suffix_b)

    #     def split_prefix(k: str):
    #         # '_fc1.lora_A' -> ('_fc1', 'lora_A')
    #         if k.endswith(suffix_a):
    #             return k[: -len(suffix_a)].rstrip("."), suffix_a
    #         if k.endswith(suffix_b):
    #             return k[: -len(suffix_b)].rstrip("."), suffix_b
    #         return None, None

    #     # 1) 先拷贝 base，并且过滤掉所有 *.sp_aggregated
    #     if clone_base:
    #         new_sd = OrderedDict(
    #             (k, (v.clone().detach() if torch.is_tensor(v) else v))
    #             for k, v in base_state_dict.items()
    #             if not has_remove_key(k)
    #         )
    #     else:
    #         new_sd = OrderedDict((k, v) for k, v in base_state_dict.items() if not has_remove_key(k))

    #     # 2) 按模板的 LoRA 键补齐零矩阵，dtype/device 优先对齐该层 weight，否则对齐模板张量
    #     added = 0
    #     for k_tmpl, v_tmpl in lora_template_state_dict.items():
    #         if not is_lora_key(k_tmpl):
    #             continue

    #         prefix, sfx = split_prefix(k_tmpl)
    #         if prefix is None:
    #             continue

    #         ref_key = f"{prefix}.weight"
    #         ref = base_state_dict.get(ref_key, v_tmpl)
    #         if not torch.is_tensor(ref):
    #             ref = v_tmpl

    #         zero_like = torch.zeros_like(v_tmpl, dtype=ref.dtype, device=ref.device)

    #         if overwrite_existing or (k_tmpl not in new_sd):
    #             new_sd[k_tmpl] = zero_like
    #             added += 1

    #     if added == 0:
    #         raise ValueError(
    #             f"模板中未发现以 '{suffix_a}' 或 '{suffix_b}' 结尾的 LoRA 参数。"
    #             f"（当前 remove_key='{remove_key}'，已移除相应 sp 缓存）"
    #         )

    #     return new_sd

    @staticmethod
    def _natural_list(s: str):
        import re
        """把字符串拆成 [文本/数字] 列表，支持 layers.10 vs layers.2 的自然排序。"""
        parts = []
        for token in s.split("."):
            # 再把 token 中的数字段拆开
            parts.extend(int(t) if t.isdigit() else t for t in re.split(r'(\d+)', token) if t != "")
        return parts

    @staticmethod
    def sort_state_dict_by_suffix(
        state_dict: dict,
        suffix_weight: str = "weight",
        suffix_bias: str   = "bias",
        suffix_a: str      = "lora_A",
        suffix_b: str      = "lora_B",
    ) -> OrderedDict:
        """
        返回一个按层前缀自然排序、且同层内按 [weight, bias, lora_A, lora_B, 其他] 排序的新 OrderedDict。
        """
        prio_map = {
            suffix_weight: 0,
            suffix_bias:   1,
            suffix_a:      2,
            suffix_b:      3,
        }

        def split_prefix_suffix(k: str):
            if "." in k:
                prefix, suf = k.rsplit(".", 1)
            else:
                prefix, suf = k, ""
            return prefix, suf

        def sort_key(k: str):
            prefix, suf = split_prefix_suffix(k)
            prio = prio_map.get(suf, 99)
            return (LoRAUtils._natural_list(prefix), prio, LoRAUtils._natural_list(suf))

        items = sorted(state_dict.items(), key=lambda kv: sort_key(kv[0]))
        return OrderedDict(items)

    @staticmethod
    def replace_weight_and_bias(
        sd1: dict,
        sd2: dict,
        *,
        suffixes=("weight", "bias"),
        cast_to_target: bool = True,   # 将 sd2 的张量转成 sd1 对应张量的 dtype/device
        strict_shape: bool = True,     # 形状不一致时抛错；设为 False 则跳过该键
        clone: bool = True             # 返回张量是否 clone().detach()
    ) -> OrderedDict:
        """
        返回: new_sd = sd1 的拷贝，其中 *.weight / *.bias 被 sd2 的对应键替换（若存在）。
        """
        new_sd = OrderedDict()
        for k, v1 in sd1.items():
            # 只匹配最后一段后缀（避免误匹配 running_mean 等）
            tail = k.rsplit(".", 1)[-1]
            should_replace = tail in suffixes and (k in sd2) \
                            and torch.is_tensor(v1) and torch.is_tensor(sd2[k])
            if should_replace:
                v2 = sd2[k]
                if strict_shape and v1.shape != v2.shape:
                    raise ValueError(f"Shape mismatch on '{k}': {v1.shape} vs {v2.shape}")
                if (not strict_shape) and (v1.shape != v2.shape):
                    # 形状不一致且允许跳过
                    new_sd[k] = v1.clone().detach() if (clone and torch.is_tensor(v1)) else v1
                    continue
                if cast_to_target:
                    v2 = v2.to(dtype=v1.dtype, device=v1.device)
                new_sd[k] = v2.clone().detach() if clone else v2
            else:
                new_sd[k] = v1.clone().detach() if (clone and torch.is_tensor(v1)) else v1
        return new_sd
