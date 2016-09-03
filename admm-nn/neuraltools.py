import numpy as np
import numpy.matlib


__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


def generate_weights(t):
    return [np.mat(np.zeros((t[i], t[i-1]), dtype='float64')) for i in range(1, len(t))]


def generate_gaussian(t, s):
    return [np.matlib.randn(t[i], s) for i in range(1, len(t))]

