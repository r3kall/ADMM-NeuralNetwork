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


def test_1():
    print()
    print("===========")
    z = np.matlib.randn(3, 4)
    a_in = np.matlib.randn(4, 4)
    w_next = np.matlib.randn(1, 3)
    z_next = np.matlib.randn(1, 4)

    w = weight_update(z, a_in)
    a = activation_update(w_next, z_next, auxiliaries.relu(z), 1, 10)

    res1 = np.dot(w, a_in)
    res2 = np.dot(w, a_in[:, 1])
    print(res1)
    print(res2)
