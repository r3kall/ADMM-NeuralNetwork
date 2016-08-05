import pytest
import numpy.matlib

import auxiliaries
from model.neuralnetwork import NeuralNetwork

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


@pytest.fixture()
def newtrain():
    nn = NeuralNetwork(1000, 10, 300)
    n = 100
    x, y = auxiliaries.data_gen(1000, 10, n)
    return nn, x, y, n


def test_train_1(newtrain):
    c = 0
    for i in range(newtrain[3]):
        newtrain[0].train(newtrain[1][i], newtrain[2][i])
        t = auxiliaries.convert_binary_to_number(newtrain[2][i])
        m, o = auxiliaries.get_max_index(newtrain[0].z[-1])
        if t == o:
            c += 1
    print("\nResult: %s/%s" % (str(c), str(newtrain[3])))