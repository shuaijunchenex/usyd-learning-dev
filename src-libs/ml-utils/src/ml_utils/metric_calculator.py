from __future__ import annotations

import torch

#TODO: unit test

class MetricCalculator:
    def __init__(self):
        None

    @staticmethod
    def matrix_norm(matrix_1, matrix_2, norm='l2'):
        diff = matrix_1 - matrix_2

        if norm == 'l2':
            if diff.ndim < 2:
                # Use vector norm
                return torch.norm(diff, p=2)
            else:
                # Use matrix norm
                return torch.linalg.matrix_norm(diff, ord=2)

        elif norm == 'frobenius':
            if diff.ndim < 2:
                return torch.norm(diff, p=2)
            else:
                return torch.linalg.matrix_norm(diff, ord='fro')

        else:
            raise ValueError(f"Unsupported norm type: {norm}")

    @staticmethod
    def weight_divergence_homogeneous_lora(wbab_1, wbab_2, norm='l2'):
        """
        Calculate the divergence for homogeneous LoRA weights.
        
        :return: Divergence value.

        TODO: finish for lora, input should be wbab data structure 
        """
        raise NotImplementedError("This method is not implemented yet.")

    @staticmethod
    def weight_divergence_normal_weight(weight_1, weight_2, norm='l2'):
        """
        Calculate the divergence between two weight vectors.
        
        :param weight_1: First weight vector.
        :param weight_2: Second weight vector.
        :return: Divergence value.
        """

        weight_divergence = 0.0

        for key in weight_1:
            weight_divergence += MetricCalculator.matrix_norm(weight_1[key], weight_2[key], norm) 

        return weight_divergence