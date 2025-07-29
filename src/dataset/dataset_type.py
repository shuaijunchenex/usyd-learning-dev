"""
定义DataSet类型枚举
"""


class EDatasetType(enumerate):
    unknown = 0

    mnist = 1
    fminst = 2
    cifar10 = 3
    cifar100 = 4
    kmnist = 5
    emnist = 6
    qmnist = 7
    svhn = 8
    stl10 = 9
    imagenet = 10
    agnews = 11
    imdb = 12
    dbpedia = 13
    yahooanswers = 14
