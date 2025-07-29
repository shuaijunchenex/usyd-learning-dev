
# Comment code for test from Nexus
# import sys
# sys.path.append("..")
# sys.path.append("../ml_data_loader")

from ml_data_loader.dataset_loader_factory import DatasetLoaderFactory
from ml_data_loader.dataset_type import EDatasetType
from ml_data_loader.dataset_loader_args import DatasetLoaderArgs

"""
Dataset loader factory unit test
"""

def main():
    args = DatasetLoaderArgs()
    args.dataset_type = EDatasetType.mnist
    args.root = "../../../.dataset"
    args.is_download = False

    print(args)

    dataset_loader = DatasetLoaderFactory.create(args)
    print(dataset_loader)
    return


if __name__ == "__main__":
    main()

