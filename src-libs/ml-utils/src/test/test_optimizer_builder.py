from __future__ import annotations
import sys
import torch


# sys.path.append("..")
# sys.path.append("../ml_utils")
# print("\n".join(sys.path))

from ml_utils.config_loader import ConfigLoader
from ml_utils.optimizer_builder import OptimizerBuilder

##############################################
# Define a simple model

model = torch.nn.Linear(10, 2)

def test_from_file():
    # Load config from yaml file
    config = ConfigLoader.load("./test_data/test_config.yaml")

    # Method 1: Load configuration from a YAML file using ConfigLoader
    optimizer = OptimizerBuilder(model.parameters(), config["optimizer"]).build()
    print("Optimizer created from configuration file:\n", optimizer)
    return

def test_direct():

    # Method 2: Provide a configuration dictionary directly
    config = {
        "optimizer": {
            "type": "Adam",
            "lr": 0.001,
            "weight_decay": 0.0001,
            "momentum": None,
            "nesterov": None,
            "betas": [0.9, 0.999],
            "amsgrad": False,
            "eps": 1e-8,
            "alpha": None,
            "centered": None
        }
    }

    optimizer = OptimizerBuilder(model.parameters(), config_dict = config["optimizer"]).build()
    print("Optimizer created from configuration dictionary:\n", optimizer)
    return


def main():
    test_from_file()
    test_direct()
    return

if __name__ == "__main__":
    main()
