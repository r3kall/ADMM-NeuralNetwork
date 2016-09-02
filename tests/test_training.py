import pytest
import time
import numpy
from sklearn import datasets

import auxiliaries
from model.neuralnetwork import NeuralNetwork

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def epoch(nn, samples, targets, tst_samples, tst_targets, n, test):
    st = time.time()
    nn = warmepochs(nn, samples, targets, 10)
    for i in range(n):
        nn.train(samples, targets)
    endt = time.time() - st
    print("Training time: %s" % numpy.round(endt, decimals=4))

    res = nn.feedforward(tst_samples)
    loss = auxiliaries.binary_loss_sum(res, tst_targets)
    print("Mean Loss: %s" % str(numpy.round(loss, decimals=4)))
    c = 0
    for i in range(test):
        output = auxiliaries.get_max_index(res[:, i])
        label = convert_to_number(tst_targets[:, i])
        if output == label:
            c += 1
    print("Accuracy: %s/%s" % (c, test))
    approx = numpy.round(float(c)/float(test), decimals=4)
    print("Approx: %s" % approx)
    print("=============")
    return nn, approx


def warmepochs(nn, samples, targets, iter):
    for i in range(iter):
        nn.warmstart(samples, targets)
    return nn


def convert_to_number(t):
    for i in range(10):
        if t[i] == 1:
            return i
    raise ValueError("Target not valid !!")


def test_1():
    print()
    print("=============")
    indim = 64
    outdim = 10
    n = 1797
    test = n
    hidden1 = 32
    nn = NeuralNetwork(indim, outdim, n, hidden1)

    """
    iris = datasets.load_iris()
    data = iris.data.T
    targets = numpy.mat(numpy.zeros((3, 150)))
    for i in range(150):
        v = iris.target[i]
        targets[v, i] = 1

    for i in range(10):
        nn = epoch(nn, data, targets, data, targets, 1, test)


    digits = datasets.load_digits()
    data = digits.data.T
    targets = numpy.mat(numpy.zeros((10, 1797)))
    for i in range(1797):
        v = digits.target[i]
        targets[v, i] = 1

    approx = 0
    it = 0
    while approx < 0.95:
        nn, approx = epoch(nn, data, targets, data, targets, 1, test)
        it += 1
    print(it)
    """