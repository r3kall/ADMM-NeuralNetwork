import pytest
import time
import numpy
from sklearn import datasets

import auxiliaries
from model.neuralnetwork import NeuralNetwork

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def epoch(nn, samples, targets, tst_samples, tst_targets, n, test):
    st = time.time()
    nn = warmepochs(nn, samples, targets, 2)
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
        label = auxiliaries.convert_triple_to_number(tst_targets[:, i])
        if output == label:
            c += 1
    print("Accuracy: %s/%s" % (c, test))
    approx = numpy.round(float(c)/float(test), decimals=4)
    print("Approx: %s" % approx)
    print("=============")
    if approx > 0.9:
        print("BINGO")
    return nn


def warmepochs(nn, samples, targets, iter):
    for i in range(iter):
        nn.warmstart(samples, targets)
    return nn


def test_1():
    print()
    print("=============")
    indim = 4
    outdim = 3
    n = 150
    test = n
    hidden1 = 15
    nn = NeuralNetwork(indim, outdim, n, hidden1)

    iris = datasets.load_iris()
    data = iris.data.T
    targets = numpy.mat(numpy.zeros((3, 150)))
    for i in range(150):
        v = iris.target[i]
        targets[v, i] = 1

    for i in range(10):
        nn = epoch(nn, data, targets, data, targets, 1, test)
