import pytest
import time
import numpy

import auxiliaries
from model.neuralnetwork import NeuralNetwork

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def generator(dim, classes):
    seed = numpy.random.randint(0, 10)
    sample = auxiliaries.sample_gen(dim, seed, 1)
    target = auxiliaries.target_gen(classes, seed)
    return sample, target, seed


@pytest.fixture(scope='module')
def train():
    in_dim = 400
    nn = NeuralNetwork(in_dim, 10, 80)
    n = 1000
    sample, target= auxiliaries.data_gen(in_dim, 10, n)
    return nn, sample, target, n


def test_train_1(train):
    nn = train[0]
    sample = train[1]
    target = train[2]
    dim = train[3]

    print("\nStart Training")
    for i in range(dim):
        nn.train(sample[i], target[i])

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
