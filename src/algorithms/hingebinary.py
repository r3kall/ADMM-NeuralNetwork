import numpy as np

from ..cyth.binarymin import binarymin

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


def bhe(z, y):
    # Binary hinge error
    if y == 1:
        return np.maximum(0, 1 - z)
    return np.maximum(0, z)
# end cost function


def argminlastz(targets, eps, w, a_in, beta):
    """
    Minimization of the last output matrix, using the above function.

    :param targets:  target matrix (equal dimensions of z)
    :param eps:      lagrange multiplier matrix (equal dimensions of z)
    :param w:        weight matrix
    :param a_in:     activation matrix l-1
    :return:         output matrix last layer
    """
    m = np.dot(w, a_in)
    # cython version
    z = binarymin(targets, eps, m, beta)
    return z
# end
