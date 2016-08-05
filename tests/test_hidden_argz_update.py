import pytest

import numpy as np
import numpy.matlib
import time

import auxiliaries
from auxiliaries import relu
from model.layers import HiddenLayer
from model.neuralnetwork import argz, NeuralNetwork, \
    weight_update, activation_update, minz

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


@pytest.fixture(scope='module')
def hidden():
    nn = NeuralNetwork(1000, 10, 300)
    a = np.matlib.randn(1000, 1)
    nn.w[0] = weight_update(nn.z[0], a)
    nn.a[0] = activation_update(nn.w[1], nn.z[1], relu(nn.z[0]), 1, 10)
    hl = HiddenLayer(1000, 300)
    return nn, a, hl


def test_hidden_argz_update_1(hidden):
    nn = hidden[0]
    a = hidden[1]
    st = time.time()
    minz(nn.z[0], nn.w[0], nn.a[0], a, relu, 1, 10)
    endt = time.time() - st
    print("\nTIME 1 (new): %s" % str(endt))


def test_hidden_argz_update_2(hidden):
    hl = hidden[2]
    a = hidden[1]
    hl.calc_weights(a)
    st = time.time()
    hl.calc_output_array(a)
    endt = time.time() - st
    print("\nTIME 2: %s" % str(endt))


