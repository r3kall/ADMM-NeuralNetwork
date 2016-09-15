import numpy as np
import scipy as sp
import scipy.optimize
import numpy.linalg

from cyth.binarymin import binarymin

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


def bhe(z, y):
    # Binary hinge error
    if y == 1:
        return np.maximum(0, 1 - z)
    return np.maximum(0, z)
# end cost function


def _minimizelast(z, y, eps, m, beta):
    """
    Minimization of z_L using the binary hinge loss function:

                  | max{1 - z, 0}, when y = 1
        l(z, y) = |
                  | max{z, 0}, when y = 0
    """

    if y == 0:
        m1 = np.maximum(0, z)
    else:
        m1 = np.maximum(0, 1 - z)
    return m1 + (z * eps) + (beta * ((z - m) ** 2))
# end


def argminlastz(targets, eps, w, a_in, beta):
    m = np.dot(w, a_in)

    x = targets.shape[0]
    y = targets.shape[1]
    z = np.mat(np.zeros((x, y)))
    for i in range(x):
        for j in range(y):
            z[i, j] = sp.optimize.minimize_scalar(_minimizelast, args=(targets[i, j], eps[i, j], m[i, j], beta)).x

    # z = binarymin(targets, eps, m, beta)
    return z
# end
