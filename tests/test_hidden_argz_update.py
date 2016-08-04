import pytest

import numpy as np
import numpy.matlib

import auxiliaries
from model.neuralnetwork import minimize, argz, argminz

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


@pytest.fixture(scope='function')
def hidden():
    a = np.matlib.randn(1000, 1)
    w = np.matlib.randn(300, 1000)
    z = np.matlib.randn(300, 1)
    act = auxiliaries.relu(np.matlib.randn(300, 1))
    res = argz(z, w, act, a, auxiliaries.relu, 1, 10)
    return z, w, act, a, res


def test_hidden_argz_update_1(hidden):
    minimize(hidden[0], hidden[4], 0.1, 2, 100, hidden[1], hidden[2], hidden[3],
             auxiliaries.relu, 1, 10)


def test_hidden_argz_update_2(hidden):
    for i in range(10):
        argminz(hidden[0], hidden[4], 0.1, 50, hidden[1], hidden[2], hidden[3],
                 auxiliaries.relu, 1, 10)


def test_hidden_argz_update_3(hidden):
    for i in range(100):
        minimize(hidden[0], hidden[4], 0.1, 2, 100, hidden[1], hidden[2], hidden[3],
                 auxiliaries.relu, 1, 10)


def test_hidden_argz_update_4(hidden):
    for i in range(1000):
        minimize(hidden[0], hidden[4], 0.1, 2, 100, hidden[1], hidden[2], hidden[3],
                 auxiliaries.relu, 1, 10)

"""
def test_hidden_argz_update_5(hidden):
    for i in range(10000):
        minimize(hidden[0], hidden[4], 0.1, 2, 100, hidden[1], hidden[2], hidden[3],
                 auxiliaries.relu, 1, 10)
"""