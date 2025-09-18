import torch
import nn_model_factory

class HetRankModelGenerator():
    def __init__(self):
        pass

    def generate_model(self, model_name: str, **kwargs) -> torch.nn.Module:
        return