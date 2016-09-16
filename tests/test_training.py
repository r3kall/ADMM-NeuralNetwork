import pytest
import time
import numpy as np

from sklearn import datasets
from neuralnetwork import NeuralNetwork, Instance
from binaryclassification import binary_classification, epoch


__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def test_digits():
    print()
    print("=============")
    dataset = datasets.load_digits(n_class=3)
    samples = dataset.data.T
    targets = np.mat(np.zeros((3, 537), dtype='uint8'))

    for i in range(537):
        v = dataset.target[i]
        targets[v, i] = 1

    trn = Instance(samples, targets)
    # nn = NeuralNetwork(1797, 64, 10, 64, 64)
    # nn, ap, r = epoch(nn, trn, trn, train_iters=0, warm_iters=128)
    # while ap < 0.95:
    #     nn, ap, r = epoch(nn, trn, trn, train_iters=1, warm_iters=0)
    nn = binary_classification(trn, trn, 64, 32, warm_iters=16, adaptive=True)


def test_iris():
    print()
    print("=============")
    from sklearn.datasets import make_moons, make_classification
    k = 300
    #X, y = make_moons(n_samples=k, noise=0.1, shuffle=False)
    X, y = make_classification(n_samples=k, scale=100.)

    targets = np.mat(np.zeros((2, k), dtype='uint8'))

    for i in range(k):
        v = y[i]
        targets[v, i] = 1

    #print(targets)
    trn = Instance(X.T, targets)
    nn = binary_classification(trn, trn, 50, warm_iters=16, adaptive=True)
