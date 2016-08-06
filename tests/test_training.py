import pytest
import time
import numpy

import auxiliaries
from model.neuralnetwork import NeuralNetwork

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def generator(dim, classes):
    seed = numpy.random.randint(0, 10)
    sample = auxiliaries.sample_gen(dim, seed)
    target = auxiliaries.target_gen(classes, seed)
    return sample, target, seed


@pytest.fixture(scope='module')
def train():
    pass


def test_train_1(train):
    in_dim = 768
    nn = NeuralNetwork(in_dim, 10, 200)
    dim = 1000
    for i in range(dim):
        sample, target, seed = generator(in_dim, 10)
        nn.train(sample, target)

    c = 0
    test = 200
    for i in range(test):
        sample, target, seed = generator(in_dim, 10)
        zl = nn.feedforward(sample, target)
        index = auxiliaries.get_max_index(zl)
        if index == float(seed):
            c += 1
    print("%s/%s" % (c, test))
