import pytest

import numpy as np
import numpy.matlib

import auxiliaries
from model.admm import weight_update, activation_update, minz

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def test_1():
    print("\n")
    a0 = np.matlib.randn(500, 1)
    w1 = np.matlib.randn(100, 500)
    w2 = np.matlib.randn(10, 100)
    a1 = np.matlib.randn(100, 1)
    a2 = np.matlib.randn(10, 1)
    z1 = np.matlib.randn(100, 1)
    z2 = np.matlib.randn(10, 1)

    # layer 1
    w1 = weight_update(z1, a0)
    a1 = activation_update(w2, z2, auxiliaries.relu(z1), 1, 2)
    z1 = minz(z1, w1, a1, a0, auxiliaries.relu, 1, 2)
    # last layer
    w2 = weight_update(z2, a1)