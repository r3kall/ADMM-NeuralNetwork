import pytest

import numpy as np
import numpy.matlib

import auxiliaries
from auxiliaries import relu
from model.layers import HiddenLayer
from model.neuralnetwork import argz, NeuralNetwork, \
    weight_update, activation_update, minz

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"




@pytest.fixture(scope='module')
def hidden():
    nn = NeuralNetwork(500, 10, 300)
    a = np.matlib.randn(500, 1)
    nn.w[0] = weight_update(nn.z[0], a)
    nn.a[0] = activation_update(nn.w[1], nn.z[1], relu(nn.z[0]), 1, 10)
    nn.w[1] = weight_update(nn.z[1], nn.a[0])
    return nn, a

def test_hidden_argz_update_1(hidden):
    nn = hidden[0]
    a = hidden[1]
    minz(nn.z[0], nn.w[0], nn.a[0], a, relu, 1, 10)
    #minz(nn.z[1], nn.w[1], nn.a[1], nn.a[0], relu, 1, 10)
