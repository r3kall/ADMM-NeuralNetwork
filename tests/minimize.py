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
    sol = ((alpha * a) + (beta * m)) / (alpha + beta)
    if a >= 0 and m >= 0:
        return sol
    elif a <= 0 and m <= 0:
        return m
    print(sol)
    print(m)
    res1 = alpha * ((a - relu(sol)) ** 2) + beta * ((relu(sol) - m) ** 2)
    res2 = alpha * (a - relu(m)) ** 2
    print(res1)
    print(res2)
    if res1 < res2:
        return sol
    return m


def test_1():
    print()
    min(0.5, -0.6, 10, 1)
