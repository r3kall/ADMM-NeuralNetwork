import math
import pytest
import time
import numpy
from sklearn import datasets

import auxiliaries
from model.neuralnetwork import NeuralNetwork

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def epoch(nn, samples, targets, tst_samples, tst_targets, train_iter=1, warm_iter=8):
    st = time.time()
    nn = warmepochs(nn, samples, targets, warm_iter)
    for i in range(train_iter):
        nn.train(samples, targets)
    endt = time.time() - st
    print("Training time: %s" % numpy.round(endt, decimals=4))

    res = nn.feedforward(tst_samples)
    test = res.shape[1]
    loss = auxiliaries.binary_loss_sum(res, tst_targets)
    print("Mean Loss: %s" % str(numpy.round(loss, decimals=4)))
    c = 0
    for i in range(test):
        output = auxiliaries.get_max_index(res[:, i])
        label = auxiliaries.convert_binary_to_number(tst_targets[:, i], 10)
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


def _minus(x, n):
    return numpy.maximum(1, x - n)


def _omega(x, minx, max):
    l = len(str(x))
    if l > 3:
        exp = int(min(math.pow(10, l-3), max))
        return int(numpy.log(x) * exp)
    return minx


def adaptive_training(nn, samples, targets, tst_samples, tst_targets, threshold=0.95, interval=5):
    i = _omega(samples.shape[1], 10, 100)
    approx = 0.
    count = 0
    diff = 0.
    t = False
    while approx < threshold:
        tmp = approx
        if count % interval == 0 and count != 0:
            if diff / interval < 0.005 and not t:
                print("boost")
                i += 100
                t = True
            diff = 0
        elif threshold - approx <= 0.01:
            i = _minus(i, i // 100)
        elif threshold - approx <= 0.1:
            i = _minus(i, i // 10)
        nn, approx = epoch(nn, samples, targets,
                           tst_samples, tst_targets, warm_iter=i)
        diff += approx - tmp
        count += 1
        print(i)
    print(count)
    return nn, approx


def meaniterations(data, targets, obj, lim, iters, b=1., g=10.):
    indim = 64
    outdim = 10
    n = 1797
    test = n
    hidden1 = 12

    it = 0
    for i in range(iters):
        nn = NeuralNetwork(indim, outdim, n, hidden1, beta=b, gamma=g)
        approx = 0
        c = 0
        while approx < obj and c < lim:
            nn, approx = epoch(nn, data, targets, data, targets)
            it += 1
            c += 1
    print("Total iterations: %s/%s" % (str(it), str(iters * lim)))
    print("Average iterations per nn: %s" % str(it / iters))
    print("=============")


def test_1():
    print()
    print("=============")
    indim = 64
    outdim = 10
    n = 1797
    test = n
    hidden1 = 12
    nn = NeuralNetwork(indim, outdim, n, hidden1)

    """
    iris = datasets.load_iris()
    data = iris.data.T
    targets = numpy.mat(numpy.zeros((3, 150)))
    for i in range(150):
        v = iris.target[i]
        targets[v, i] = 1
    """

    digits = datasets.load_digits()
    data = digits.data.T
    targets = numpy.mat(numpy.zeros((10, 1797)))
    for i in range(1797):
        v = digits.target[i]
        targets[v, i] = 1

    for i in range(1):
        nn, approx = adaptive_training(nn, data, targets, data, targets)
