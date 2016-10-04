#import pytest
import numpy as np

from sklearn import datasets
from src.neuralnetwork import NeuralNetwork, Instance
from binaryclassification import binary_classification


__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def check(set, dim, k=3):
    from commons import convert_binary_to_number
    count = [0 for i in range(dim)]
    for j in range(set.shape[1]):
        count[convert_binary_to_number(set[:, j], dim)] += 1
    mean = sum(count) / dim
    minz = min(count)
    maxz = max(count)
    if minz + k < mean < maxz - k:
        return False
    return True


def test_digits():
    print()
    print("=============")
    dataset = datasets.load_digits()
    samples = dataset.data.T
    targets = np.mat(np.zeros((10, 1797), dtype='uint8'))

    for i in range(1797):
        v = dataset.target[i]
        targets[v, i] = 1

    ist = Instance(samples, targets)
    nn = binary_classification(ist, ist, 64, warm_iters=16, adaptive=True)


def test_iris():
    print()
    print("=============")
    from sklearn.datasets import make_moons, make_classification
    from neuraltools import split_instance
    k = 100
    X, y = make_moons(n_samples=k, noise=0.1, shuffle=False)
    # X, y = make_classification(n_samples=k, scale=10.)

    targets = np.mat(np.zeros((2, k), dtype='uint8'))

    for i in range(k):
        v = y[i]
        targets[v, i] = 1

    ist = Instance(X.T, targets)
    trn, tst = split_instance(ist, percentage=50, shuffle=False)
    flag = check(tst.targets, 2, k=0)
    c = 0
    while not flag:
        trn, tst = split_instance(ist, percentage=50, shuffle=True)
        flag = check(tst.targets, 2, k=0)
        c += 1
    print(c)

    nn = binary_classification(trn, tst, 75, warm_iters=2, adaptive=False)
