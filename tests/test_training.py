import pytest
import time
import numpy
import profile

import auxiliaries
from model.neuralnetwork import NeuralNetwork

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def test_1():
    print()
    indim = 768
    outdim = 10
    n = 512
    hidden1 = 200
    nn = NeuralNetwork(indim, outdim, n, hidden1)

    for i in range(20):
        samples, targets = auxiliaries.data_gen(indim, outdim, n)
        print(nn.w[-1][0, 0])
        print("===============")
        nn.train(samples, targets)


    test = n // 2
    samples, targets = auxiliaries.data_gen(indim, outdim, test)
    res = nn.feedforward(samples)

    c = 0
    for i in range(test):
        output = auxiliaries.get_max_index(res[:, i])
        label = auxiliaries.convert_binary_to_number(targets[:, i])
        if output == label:
            c += 1
    print("%s/%s" % (c, test))

