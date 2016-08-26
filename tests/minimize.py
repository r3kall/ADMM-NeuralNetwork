from cmath import sqrt

import pytest

import numpy as np
import scipy as sp
import numpy.matlib
import scipy.optimize
import time

import auxiliaries

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def relu(x):
    return np.maximum(0, x)


def min(a, m, alpha, beta):
    if a <= 0 and m <= 0:
        return m
    sol = ((alpha * a) + (beta * m)) / (alpha + beta)
    if a >= 0 and m >= 0:
        return sol
    if m < 0 < a:
        if sol / (a ** 2) > 1:
            return sol
        else:
            return m
    if a < 0 < m:
        return sol


def test_1():
    print()
    for i in range(1000*10000):
        res = min(0.5, -0.5, 10, 1)
