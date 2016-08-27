import random

import pytest

import numpy as np
import scipy as sp
import numpy.matlib
import scipy.linalg.blas
import time

import auxiliaries
from model.admm import weight_update, activation_update, argminz, argminlastz

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def test_weight():
    print()
    n = 1002
    z1 = np.matlib.randn(300, n)
    a0 = np.matlib.randn(1000, n)

    #w1 = weight_update(z1, a0)
    #print("Weight shape: %s" % str(w1.shape))


def test_activation():
    print()
    indim = 2048
    outdim = 512
    n = 1024
    w1 = np.matlib.randn(outdim, indim)
    z1 = np.matlib.randn(outdim, n)
    a0 = auxiliaries.relu(np.matlib.randn(indim, n))

    st = time.time()
    #a0 = activation_update(w1, z1, a0, 1, 10)
    C1 = np.dot(w1.H, w1)
    #C2 = sp.linalg.blas.dgemm(alpha=1.0, a=w1.H, b=w1)
    endt = time.time() - st
    print("Activation shape: %s" % str(a0.shape))
    print(endt)


def test_minz():
    print()
    n = 2
    indim = 300
    outdim = 3
    w = np.matlib.randn(outdim, indim)
    #z = np.matlib.randn(outdim, n)
    act = np.matlib.randn(outdim, n)
    a = np.matlib.randn(indim, n)

    z = argminz(act, w, a, 10, 1)
    print("Output shape: %s" % str(z.shape))
    print()
    print(z)


def _comp(z, y, eps, m, beta):
    c = 0
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            c += auxiliaries.binary_hinge_loss(z[i, j], y[i, j]) + (z[i, j] * eps[i, j]) + \
                 beta * ((z[i, j] - m[i, j]) ** 2)
    return c


def test_minlastz():
    print()
    n = 2
    indim = 128
    outdim = 10
    w = np.matlib.randn(outdim, indim)
    z = np.matlib.randn(outdim, n)
    eps = np.matlib.randn(outdim, n)
    a_in = np.matlib.randn(indim, n)
    m = np.dot(w, a_in)

    samples, targets = auxiliaries.data_gen(indim, outdim, n)

    print(_comp(z, targets, eps, m, 1.))
    print(z)
    print("==========")
    z = argminlastz(targets, eps, w, a_in, 1.)
    print(_comp(z, targets, eps, m, 1.))
    print(z)


