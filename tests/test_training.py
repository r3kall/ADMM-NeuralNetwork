import pytest
import time
import numpy

import auxiliaries
from model.neuralnetwork import NeuralNetwork

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


@pytest.fixture(scope='module')
def train():
    in_dim = 100
    n = 256
    nn = NeuralNetwork(in_dim, 10, n, 25)
    return nn, in_dim, n


def test_train_1(train):
    nn = train[0]
    in_dim = train[1]
    n = train[2]
    iter = 32
    samples, targets = auxiliaries.data_gen(in_dim, 10, n)
    print("\nStart Training")
    for i in range(iter):
        print("Iter: %s" % str(i+1))
        st = time.time()
        nn.train(samples, targets)
        endt = time.time() - st
        print("Time: %s" % str(round(endt, 4)))

    c = 0
    test = 82
    samples, targets = auxiliaries.data_gen(in_dim, 10, test)
    print("Start Testing")
    res = nn.feedforward(samples)
    for i in range(test):
        y = auxiliaries.convert_binary_to_number(targets[:, i])
        x = auxiliaries.get_max_index(res[:, i])
        if float(x) == y:
            c += 1
    print("%s/%s" % (c, test))
