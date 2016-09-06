import pytest
import math
import time
import numpy as np
from copy import deepcopy
from sklearn import datasets

import commons
from data_processing import Mnist
from functions import mbhe
from neuralnetwork import NeuralNetwork
from neuraltools import save_network_to_file, load_network_from_file

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def epoch(nn, samples, targets, tst_samples, tst_targets, train_iter=1, warm_iter=4):
    st = time.time()
    nn = warmepochs(nn, samples, targets, warm_iter)
    for i in range(train_iter):
        nn.train(samples, targets)
    endt = time.time() - st
    print("Training time: %s" % np.round(endt, decimals=4))

    res = nn.feedforward(tst_samples)
    test = res.shape[1]
    loss = mbhe(res, tst_targets)
    print("Mean Loss: %s" % str(np.round(loss, decimals=4)))
    c = 0
    for i in range(test):
        output = commons.get_max_index(res[:, i])
        label = commons.convert_binary_to_number(tst_targets[:, i], 10)
        if output == label:
            c += 1
    print("Accuracy: %s/%s" % (c, test))
    approx = np.round(float(c)/float(test), decimals=4)
    print("Approx: %s" % approx)
    resid = np.round(residual(nn.z[-1], nn.w[-1], nn.a[-2], nn.beta), decimals=6)
    print("Residual: %s" % str(resid))
    print("=============")
    return nn, approx


def warmepochs(nn, samples, targets, iter):
    for i in range(iter):
        nn.warmstart(samples, targets)
    return nn


def residual(z, w, a_in, beta):
    m = beta * np.mat((z - np.dot(w, a_in)))
    rows = m.shape[0]
    cols = m.shape[1]
    c = 0
    for i in range(rows):
        for j in range(cols):
            c += np.abs(m[i, j])
    return c / (rows * cols)


def _minus(x, n):
    return np.maximum(1, x - n)


def _omega(x, minx, max):
    l = len(str(x))
    if l > 3:
        exp = int(min(math.pow(10, l-3), max))
        return int(np.log(x) * exp)
    return minx


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


def find_params(nn, samples, targets, tst_samples, tst_targets, eps):
    nn0, approx = epoch(deepcopy(nn), samples, targets,
                        tst_samples, tst_targets, warm_iter=1)
    count = 10
    gamma = nn.gamma
    v = 0
    while count > 0 and gamma - eps > 0:
        if v >= 0:
            nn.gamma = gamma + eps
            nnp, p = epoch(deepcopy(nn), samples, targets,
                           tst_samples, tst_targets, warm_iter=1)
        if v <= 0:
            nn.gamma = gamma - eps
            nnt, t = epoch(deepcopy(nn), samples, targets,
                           tst_samples, tst_targets, warm_iter=1)
        if approx >= p and approx >= t:
            return gamma
        if p > t:
            gamma = gamma + eps
            approx = p
            v = 1
        else:
            gamma = gamma - eps
            approx = t
            v = -1
        count -= 1
    return gamma


def test_mnist():
    print()
    print("=============")
    trnset = Mnist.getTrainingSet()
    tstset = Mnist.getTestingSet()
    trnx = trnset['x']
    trny = trnset['y']
    tstx = tstset['x']
    tsty = tstset['y']

    nn = NeuralNetwork(trnx.shape[1], trnx.shape[0], trny.shape[0], 300)
    nn, approx = epoch(nn, trnx, trny, tstx, tsty, train_iter=0, warm_iter=5)
    while approx < 0.8:
        nn, approx = epoch(nn, trnx, trny, tstx, tsty, train_iter=1, warm_iter=4)
    save_network_to_file(nn)


def test_digits():
    print()
    print("=============")
    dataset = datasets.load_digits()
    samples = dataset.data.T
    labels = dataset.target
    targets = np.mat(np.zeros((10, len(labels)), dtype=np.uint8))
    for i in range(len(labels)):
        v = labels[i]
        targets[v, i] = 1

    nn = NeuralNetwork(1797, 64, 10, 100)

    nn, approx = epoch(nn, samples, targets, samples, targets,
                              train_iter=1, warm_iter=10)
    c = 0
    while approx < 0.95:
        c += 1
        time.sleep(1)
        nn, approx = epoch(nn, samples, targets, samples, targets,
                                  train_iter=1, warm_iter=2)
    print(c)
