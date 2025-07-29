import torch
import torch.nn as nn
import copy
import torch.nn.init as init
import math
# import sys
# sys.path.insert(0, '')
from lora.lora_implementation.lora_linear import LoRALinear
from model_extractor.advanced_model_extractor import AdvancedModelExtractor

class MatrixApproximator:

    def __init__(self, base_model, use_sqrt = True, rank = 8, device = "cpu"):
        self.base_model = base_model
        self.use_sqrt = use_sqrt
        self.rank = rank
        self.device = device

    def simple_lora_model_generator(self, a_init_method='xavier', b_init_method='zeros'):
        model_copy = copy.deepcopy(self.base_model)
        self._replace_module_with_init_lora(model_copy, a_init_method, b_init_method)

        extractor = AdvancedModelExtractor(model_copy)
        wbab = extractor._extract_all_layers()

        return model_copy, wbab

    def approximate_lora_model(self):
        model_copy = copy.deepcopy(self.base_model)
        self._replace_module_with_approximation(model_copy)

        extractor = AdvancedModelExtractor(model_copy)
        wbab = extractor._extract_all_layers()

        return model_copy, wbab

    def _init_tensor(self, tensor, method):
        if method == "xavier":
            init.xavier_uniform_(tensor)
        elif method == "kaiming":
            init.kaiming_uniform_(tensor, a=math.sqrt(5))
        elif method == "normal":
            init.normal_(tensor, mean=0.0, std=0.02)
        elif method == "zeros":
            init.zeros_(tensor)
        elif method == "ones":
            init.ones_(tensor)
        else:
            raise ValueError(f"Unsupported init method: {method}")

    def _replace_module_with_init_lora(self, module, a_init_method, b_init_method):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                W = child.weight.data.to(self.device)
                bias_flag = child.bias is not None

                lora_layer = LoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    rank=self.rank,
                    use_bias=bias_flag,
                    pretrained_weight=W,
                    device=self.device
                )

                with torch.no_grad():
                    self._init_tensor(lora_layer.lora_A, a_init_method)
                    self._init_tensor(lora_layer.lora_B, b_init_method)
                    if bias_flag:
                        lora_layer.bias.copy_(child.bias.data)

                setattr(module, name, lora_layer)
            else:
                self._replace_module_with_init_lora(child, a_init_method, b_init_method)

    def _replace_module_with_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                W = child.weight.data.to(self.device)
                bias_flag = child.bias is not None

                A, B = (
                    self.sqrt_approximation(W, self.rank)
                    if self.use_sqrt else
                    self.regular_approximation(W, self.rank)
                )

                lora_layer = LoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    rank=self.rank,
                    use_bias=bias_flag,
                    pretrained_weight=W,
                    device=self.device,
                    scaling = 1
                )

                with torch.no_grad():
                    lora_layer.lora_A.copy_(A)
                    lora_layer.lora_B.copy_(B)
                    if bias_flag:
                        lora_layer.bias.copy_(child.bias.data)

                setattr(module, name, lora_layer)
            else:
                self._replace_module_with_approximation(child)


    @staticmethod
    def sqrt_approximation(W: torch.Tensor, rank: int):
        """
        Decomposes the matrix W into two matrices A and B such that A @ B approximates W,
        using truncated SVD.

        Args:
            W (torch.Tensor): The original matrix of shape (m, n).
            rank (int): The target rank for approximation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                A (m, rank), B (rank, n) such that W ≈ A @ B
        """
        # SVD decomposition
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # Vh is V^T

        # Truncate to rank
        U_r = U[:, :rank]                    # (m, r)
        S_r = S[:rank]                       # (r,)
        Vh_r = Vh[:rank, :]                  # (r, n)

        # Compute square root of singular values
        S_root = torch.sqrt(S_r)            # (r,)

        # Construct A and B
        A = U_r @ torch.diag(S_root)        # (m, r)
        B = torch.diag(S_root) @ Vh_r       # (r, n)

        return A, B

    @staticmethod
    def regular_approximation(W: torch.Tensor, rank: int):
        """
        Decomposes the matrix W into two matrices A and B such that A @ B approximates W,
        using truncated SVD.

        Args:
            W (torch.Tensor): The original matrix of shape (m, n).
            rank (int): The target rank for approximation.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                A (m, rank), B (rank, n) such that W ≈ A @ B
        """
        # SVD decomposition
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # Vh is V^T

        # Truncate to rank
        U_r = U[:, :rank]             # (m, r)
        S_r = S[:rank]                # (r,)
        Vh_r = Vh[:rank, :]           # (r, n)

        # Fix: Turn S into a diagonal matrix
        S_diag = torch.diag(S_r)     # (r, r)

        # Construct A and B
        A = U_r @ S_diag             # (m, r)
        B = Vh_r                     # (r, n)

        return A, B

# Test function
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '')
    from model_extractor.advanced_model_extractor import AdvancedModelExtractor

    def test_matrix_approximator():
        torch.manual_seed(42)

        # Test: matrix-level approximation
        m, n = 64, 128
        rank = 16
        W = torch.randn(m, n)

        A_sqrt, B_sqrt = MatrixApproximator.sqrt_approximation(W, rank)
        W_sqrt_approx = A_sqrt @ B_sqrt
        sqrt_error = torch.norm(W - W_sqrt_approx) / torch.norm(W)

        A_reg, B_reg = MatrixApproximator.regular_approximation(W, rank)
        W_reg_approx = A_reg @ B_reg
        reg_error = torch.norm(W - W_reg_approx) / torch.norm(W)

        print("Original matrix shape:", W.shape)
        print(f"Target rank: {rank}")
        print(f"[sqrt_approximation] Relative Frobenius Error: {sqrt_error:.6f}")
        print(f"[regular_approximation] Relative Frobenius Error: {reg_error:.6f}")
        assert A_sqrt.shape == (m, rank)
        assert B_sqrt.shape == (rank, n)
        assert A_reg.shape == (m, rank)
        assert B_reg.shape == (rank, n)
        print("Matrix shape assertions passed.")

        # Test: approximate a full model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(20, 10)
                self.linear2 = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear2(torch.relu(self.linear1(x)))

        model = DummyModel()
        approximator = MatrixApproximator(model, rank=4, use_sqrt=True)
        lora_model = approximator.approximate_lora_model()

        w_b_AB = AdvancedModelExtractor(lora_model)._extract_all_layers()

        # Verify that all linear layers have been replaced
        linear_count = sum(1 for m in lora_model.modules() if isinstance(m, nn.Linear))
        lora_count = sum(1 for m in lora_model.modules() if isinstance(m, LoRALinear))
        print(f"Replaced {lora_count} Linear layers with LoRALinear (expected 2).")
        assert linear_count == 0 and lora_count == 2
        print("Model layer replacement verified.")

    test_matrix_approximator()

