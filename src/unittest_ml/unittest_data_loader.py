from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from usyd_learning.ml_utils import console, ConfigLoader
from usyd_learning.ml_data_loader import DatasetLoaderFactory, DatasetLoaderArgs


def main():
    yaml_file_name = './test_data/node_config_template_client.yaml'

    # Form yaml file
    console.out(f"Test from yaml file: {yaml_file_name}")
    console.out("------------- Begin ---------------")

    yaml = ConfigLoader.load(yaml_file_name)
    console.out(yaml)

    args = DatasetLoaderFactory.create_args(yaml)
    args.root = "../../../.dataset"
    args.is_download = False

    print(args)

    dataset_loader = DatasetLoaderFactory.create(args)
    print(dataset_loader)

    console.out("------------- End -----------------")
    return


if __name__ == "__main__":
    main()

