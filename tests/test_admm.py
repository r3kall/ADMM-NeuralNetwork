import pytest

import numpy as np
import numpy.matlib

import auxiliaries
from model.admm import weight_update, activation_update, minz, minlastz, argz, lambda_update

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
    n = 102
    indim = 300
    outdim = 80
    w = np.matlib.randn(outdim, indim)
    z = np.matlib.randn(outdim, n)
    act = np.matlib.randn(outdim, n)
    a = np.matlib.randn(indim, n)

    z = minz(z, w, act, a, auxiliaries.relu, 1, 10)
    print("Output shape: %s" % str(z.shape))


def test_minlastz():
    print()
    n = 12
    w = np.matlib.randn(10, 400)
    z = np.matlib.randn(10, n)
    a = np.matlib.randn(400, n)
    l = np.matlib.randn(10, n)
    y = np.matlib.randn(10, n)
    mp = np.dot(w, a)

    res = minlastz(z, y, auxiliaries.quadratic_cost, l, mp, z, 1)
    print("Final shape: %s" % str(res.shape))
    lmb = lambda_update(res, mp, 1)
    print("Lambda shape: %s" % str(lmb.shape))

