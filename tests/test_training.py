import pytest
import time
import numpy

import auxiliaries
from model.neuralnetwork import NeuralNetwork

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


@pytest.fixture(scope='module')
def train():
    in_dim = 400
    n = 12
    nn = NeuralNetwork(in_dim, 10, n, 80)
    sample, target = auxiliaries.data_gen(in_dim, 10, n)
    return nn, sample, target, n


def test_train_1(train):
    nn = train[0]
    sample = train[1]
    target = train[2]
    dim = train[3]

    print("\nStart Training")
    nn.train(sample, target)

    """
    c = 0
    test = 100
    print("Start Testing")
    for i in range(test):
        zl = nn.feedforward(sample[i])
        #print(zl)
        index = auxiliaries.get_max_index(zl)
        y = auxiliaries.convert_binary_to_number(target[i])
        #print(y)
        if index == y:
            c += 1
    print("%s/%s" % (c, test))
    """
