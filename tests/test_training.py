import pytest
import time
import numpy
import profile

import auxiliaries
from model.neuralnetwork import NeuralNetwork

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def epoch(nn, samples, targets, tst_samples, tst_targets, n, test):
    st = time.time()
    for i in range(n):
        nn.train(samples, targets)
    endt = time.time() - st
    print("Training time: %s" % numpy.round(endt, decimals=4))

    res = nn.feedforward2(tst_samples)
    c = 0
    for i in range(test):
        output = auxiliaries.get_max_index(res[:, i])
        label = auxiliaries.convert_binary_to_number(tst_targets[:, i])
        if output == label:
            c += 1
    print("Accuracy: %s/%s" % (c, test))
    approx = numpy.round(float(c)/float(test), decimals=4)
    print("Approx: %s" % approx)
    print("===========")
    if approx > 0.5:
        print("BINGO")
        exit(0)
    return nn


def warmepochs(nn, samples, targets, iter):
    for i in range(iter):
        nn.warmstart(samples, targets)
    return nn


def test_1():
    print()
    indim = 768
    outdim = 10
    n = 512
    test = n // 6
    hidden1 = 100
    hidden2 = 50
    nn = NeuralNetwork(indim, outdim, n, hidden1, hidden2)

    samples, targets = auxiliaries.data_gen(indim, outdim, n)
    tst_samples, tst_targets = auxiliaries.data_gen(indim, outdim, test)
    nn = warmepochs(nn, samples, targets, 5)
    for i in range(100):
        nn = epoch(nn, samples, targets, tst_samples, tst_targets, 5, test)
        #m = numpy.dot(nn.w[-1], nn.a[-2])
        #print(nn.z[-1][0, 0])
        #print(m[0, 0])
        #print(nn.lAmbda[0, 0])
