import pytest
import numpy.matlib

import auxiliaries
from model.fnn import FNN
from model.neuralnetwork import NeuralNetwork

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


@pytest.fixture()
def newtrain():
    nn = NeuralNetwork(1000, 10, 300)
    # x, y = auxiliaries.data_gen(1000, 10, 10)
    x = numpy.matlib.randn(1000, 1)
    y = numpy.matlib.randn(10, 1)
    return nn, x, y

def test_train_1(newtrain):
    newtrain[0].train(newtrain[1], 0)


@pytest.fixture()
def oldtrain():
    nn = FNN(1000, 10, 300)
    x, y = auxiliaries.data_gen(1000, 10, 10)
    return nn, x, y

@pytest.mark.usefixtures("oldtrain")
def test_train_2(oldtrain):
    nn = oldtrain[0]
    x = oldtrain[1]
    y = oldtrain[2]
    nn.train(x, y, 10)
    nn.validate(x, y, 10)
