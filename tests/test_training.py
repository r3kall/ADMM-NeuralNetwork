import pytest
import time

import auxiliaries
from model.neuralnetwork import NeuralNetwork

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


@pytest.fixture()
def train():
    nn = NeuralNetwork(1000, 10, 300)
    n = 10
    x, y = auxiliaries.data_gen(1000, 10, n)
    return nn, x, y, n


def test_train_1(train):
    st = time.time()
    c = 0
    for i in range(train[3]):
        train[0].train(train[1][i], train[2][i])
        t = auxiliaries.convert_binary_to_number(train[2][i])
        m, o = auxiliaries.get_max_index(train[0].z[-1])
        if t == o:
            c += 1
    endt = time.time() - st
    print("\nResult: %s/%s" % (str(c), str(train[3])))
    print("Time: %s" % str(endt))
