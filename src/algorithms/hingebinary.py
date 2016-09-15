import numpy as np

from cyth.binarymin import binarymin

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


def bhe(z, y):
    # Binary hinge error
    if y == 1:
        return np.maximum(0, 1 - z)
    return np.maximum(0, z)
# end cost function


def argminlastz(targets, eps, w, a_in, beta):
    m = np.dot(w, a_in)
    z = binarymin(targets, eps, m, beta)
    return z
# end
