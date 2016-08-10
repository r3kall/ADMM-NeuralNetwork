import pytest
import time
import numpy
import profile

import auxiliaries
from model.neuralnetwork import NeuralNetwork

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


@pytest.fixture(scope='module')
def train():
    in_dim = 256
    n = 64
    nn = NeuralNetwork(in_dim, 10, n, 64)
    return nn, in_dim, n


def test_train_1(train):
    print()
    nn = train[0]
    in_dim = train[1]
    n = train[2]
    warmstart = 1
    iter = 2
    samples, targets = auxiliaries.data_gen(in_dim, 10, n)
    print("\nStart Warm Start Training")
    for i in range(warmstart):
        print("Warm Iter: %s" % str(i+1))
        st = time.time()
        nn.warmstart(samples, targets)
        endt = time.time() - st
        print("Warmstart Time: %s" % str(round(endt, 4)))
    print("\nStart Training")
    for i in range(iter):
        print("Iter: %s" % str(i+1))
        st = time.time()
        nn.train(samples, targets)
        endt = time.time() - st
        print("Time: %s" % str(round(endt, 4)))

    c = 0
    test = 16
    samples, targets = auxiliaries.data_gen(in_dim, 10, test)
    print("Start Testing")
    res = nn.feedforward(samples)

    for i in range(test):
        y = auxiliaries.convert_binary_to_number(targets[:, i])
        x = auxiliaries.get_max_index(res[:, i])
        #print(res[:, i])
        #print("X: %s" % str(x))
        #print("Y: %s" % str(y))
        if float(x) == y:
            c += 1
    print("1: %s/%s" % (c, test))
