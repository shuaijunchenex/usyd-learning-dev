from enum import StrEnum

"""
定义DataSet类型枚举
"""

class EDatasetType(StrEnum):
    unknown = "unknown"

    mnist = "mnist"
    fminst = "fmnist"
    cifar10 = "cifar10"
    cifar100 = "cifar100"
    kmnist = "kmnist"
    emnist = "emnist"
    qmnist = "qmnist"
    svhn = "svhn"
    stl10 = "stl10"
    imagenet = "imagenet"
    agnews = "agnews"
    imdb = "imdb"
    dbpedia = "dbpedia"
    yahooanswers = "yahooanswers"
