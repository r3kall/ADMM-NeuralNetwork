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

    samples, targets = auxiliaries.data_gen(indim, outdim, n)
    nn.train(samples, targets)

    c = 0
    print(nn.z[-1].shape)
    for j in range(n):
        output = auxiliaries.get_max_index(nn.z[-1][:, j])
        label = auxiliaries.convert_binary_to_number(targets[:, j])
        if output == label:
            c += 1
    print("===")
    print(c)

    """
    test = n // 6
    samples, targets = auxiliaries.data_gen(indim, outdim, test)
    res = nn.feedforward(samples)

    c = 0
    for i in range(test):
        output = auxiliaries.get_max_index(res[:, i])
        label = auxiliaries.convert_binary_to_number(targets[:, i])
        if output == label:
            c += 1
    print("%s/%s" % (c, test))
    """
