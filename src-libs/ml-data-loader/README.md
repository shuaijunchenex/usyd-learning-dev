# ml-data-loader Package

The 'ml-data-loader' package provide encapsulated classes to load ML dataset, support dataset as follows:

  -  mnist
  -  fminst
  -  cifar10
  -  cifar100
  -  kmnist
  -  emnist
  -  qmnist
  -  svhn
  -  stl10
  -  imagenet
  -  agnews
  -  imdb
  -  dbpedia
  -  yahooanswers



## Usage

```python
... ...
args = DataLoadArgs()					   #data loader args
args.dataset_type = EDatasetType.mnist		#specifiy dataset type
dataset_loader = DatasetLoaderFactory.create(self.args)		#create data loader
... ...
```

## Package build & deploy

Step 1: Change deploy VERSION in pyproject.toml

```
###
# Package version
###
version = "0.0.3"			# here
```

Step 2: Build package command:
```cmd
python -m build
```
Step 3: Upload package to Nexus command:
```cmd
twine upload -r nexus dist/*
```

Step 4: Pip install or upgrade

```
pip install ml_data_loader
pip install ml_data_loader --upgrade
```

