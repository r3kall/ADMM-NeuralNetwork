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
    n = 4096
    hidden1 = 300
    hidden2 = 100
    nn = NeuralNetwork(indim, outdim, n, hidden1, hidden2)

    samples, targets = auxiliaries.data_gen(indim, outdim, n)
    print("Start Warmstart")
    for i in range(6):
        #print(nn.w[-1][0, 0])
        #print("===============")
        nn.warmstart(samples, targets)
    print("End Warmstart")
    print("Start Training")
    for i in range(8):
        nn.train(samples, targets)
    print("End Training")

    test = n // 2
    samples, targets = auxiliaries.data_gen(indim, outdim, test)
    res = nn.feedforward2(samples)

    c = 0
    #print(res[:, 0])
    for i in range(test):
        output = auxiliaries.get_max_index(res[:, i])
        #print(output)
        label = auxiliaries.convert_binary_to_number(targets[:, i])
        #print(label)
        #print("=========")

        if output == label:
            c += 1
    print("%s/%s" % (c, test))

