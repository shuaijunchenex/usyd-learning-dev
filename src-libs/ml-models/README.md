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

Build package command:
```cmd
python -m build
```
Upload package to Nexus command:
```cmd
twine upload -r nexus dist/*
```
