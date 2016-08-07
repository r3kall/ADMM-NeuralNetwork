import pytest

import numpy as np
import numpy.matlib

import auxiliaries
from model.admm import weight_update, activation_update, minz, minlastz, argz

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def test_weight():
    print()
    n = 1002
    z1 = np.matlib.randn(300, n)
    a0 = np.matlib.randn(1000, n)

    w1 = weight_update(z1, a0)
    print("Weight shape: %s" % str(w1.shape))


def test_activation():
    print()
    n = 1002
    w1 = np.matlib.randn(300, 1000)
    z1 = np.matlib.randn(300, n)
    a0 = auxiliaries.relu(np.matlib.randn(1000, n))

    a0 = activation_update(w1, z1, a0, 1, 10)
    print("Activation shape: %s" % str(a0.shape))


def test_minz():
    print()
    n = 10
    w = np.matlib.randn(300, 1000)
    z = np.matlib.randn(300, n)
    act = np.matlib.randn(300, n)
    a = np.matlib.randn(1000, n)

    z = minz(z, w, act, a, auxiliaries.relu, 1, 10)
    #print("Output shape: %s" % str(z.shape))
