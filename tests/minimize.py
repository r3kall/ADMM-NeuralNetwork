from cmath import sqrt

import pytest

import numpy as np
import scipy as sp
import numpy.matlib
import scipy.optimize
import time

import auxiliaries

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def _minimize(a, m, alpha, beta):
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


def argminz(z, a, w, a_in, gamma, beta):
    m = np.dot(w, a_in)
    x = z.shape[0]
    y = z.shape[1]
    #z = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            z[i, j] = _minimize(a[i, j], m[i, j], gamma, beta)
    return z


def comp(z, a, w, a_in, gamma, beta):
    m = np.dot(w, a_in)
    m1 = gamma * (np.linalg.norm(a - np.maximum(0, z))) ** 2
    m2 = beta * (np.linalg.norm(z - m)) ** 2
    return m1 + m2


def test_1():
    print()
    n = 3
    indim = 400
    outdim = 3
    w = np.matlib.randn(outdim, indim)
    z = np.matlib.randn(outdim, n)
    a = np.matlib.randn(outdim, n)
    a_in = np.matlib.randn(indim, n)
    m = np.dot(w, a_in)
    res = comp(z, a, w, a_in, 10, 1)
    print("=======")
    print(a)
    print("----------------------------------------------")
    print(m)
    print("=======")
    print("Original score: %s" % str(res))
    print("=======")
    z = argminz(z, a, w, a_in, 10, 1)
    res = comp(z, a, w, a_in, 10, 1)
    print(res)
    print(z)


def test_2():
    print()
    res = _minimize(2.5, -41.5, 10, 1)
    res1 = 10 * (2.5) ** 2
    sol = np.abs(((10 * 2.5) + (1 * -41.5)) / (11))
    res2 = 10 * (2.5 - np.maximum(0, sol)) ** 2 + 1 * (sol + 41.5) ** 2
    print(res1)
    print(res2)