import sys

sys.path.insert(0, "..")
sys.path.insert(0, "../ml_models")

from ml_models.nn_model_args import NNModelArgs
from ml_models.nn_model_factory import NNModelFactory
from ml_models.nn_model_type import ENNModelType

def main():
    args = NNModelArgs()    
    args.is_download = True
    print(args)

    args.model_type = ENNModelType.mnist2NNBrenden
    nn_model = NNModelFactory.create(args)

    print(nn_model)



if __name__ == "__main__":
    main()
