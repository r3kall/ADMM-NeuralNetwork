import pytest
import time
import numpy as np

from sklearn import datasets
from neuralnetwork import NeuralNetwork, Instance
from binaryclassification import binary_classification, find_params, epoch


__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"

"""
def adaptive_training(nn, samples, targets, tst_samples, tst_targets, threshold=0.95, interval=5):
    i = np.minimum(_omega(samples.shape[1], 10, 100), 100)
    print(i)
    approx = 0.
    count = 0
    diff = 0.
    t = False
    while approx < threshold:
        tmp = approx
        if count % interval == 0 and count != 0:
            if diff / interval < 0.005 and not t:
                print("boost")
                i += np.minimum(_omega(samples.shape[1], 10, 100), 100)
                t = True
            diff = 0
        else:
            i = _minus(i, i // 20)
        nn, approx = epoch(nn, samples, targets,
                           tst_samples, tst_targets, warm_iter=i, train_iter=1)
        diff += approx - tmp
        count += 1
        print(i)
    print(count)
    return nn, approx
"""


def test_digits():
    print()
    print("=============")
    dataset = datasets.load_digits()
    samples = dataset.data.T
    targets = np.mat(np.zeros((10, 1797), dtype='uint8'))
    for i in range(1797):
        v = dataset.target[i]
        targets[v, i] = 1

    trn = Instance(samples, targets)
    nn = NeuralNetwork(1797, 64, 10, 128, 64)
    # nn, ap = epoch(nn, trn, trn, train_iters=0, warm_iters=4)
    # while ap < 0.95:
    #     nn, ap = epoch(nn, trn, trn, train_iters=1, warm_iters=0)
    nn = binary_classification(None, trn, trn, 128, warm_iters=4)
