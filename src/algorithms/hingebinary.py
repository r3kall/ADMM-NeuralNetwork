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


def _minimizelast(y, eps, m, beta):
    """
    Minimization of z_L using the binary hinge loss function:

                  | max{1 - z, 0}, when y = 1
        l(z, y) = |
                  | max{z, 0}, when y = 0
    """

    if y == 0:
        if m >= ((1 + eps) / (2 * beta)):
            return m - ((1 + eps) / (2 * beta))
        else:
            return m - (eps / (2 * beta))
    else:
        sol = m + ((1 - eps) / (2 * beta))
        if sol <= 1:
            return sol
        else:
            return m - (eps / (2 * beta))
# end


def argminlastz(targets, eps, w, a_in, beta):
    m = np.dot(w, a_in)
    """
    x = targets.shape[0]
    y = targets.shape[1]
    z = np.mat(np.zeros((x, y), dtype='float64'))
    for i in range(x):
        for j in range(y):
            z[i, j] = _minimizelast(targets[i, j], eps[i, j], m[i, j], beta)
    """
    z = binarymin(targets, eps, m, beta)
    return z
# end


# my minimizer is better !! (really)
def testfoo(z, targets, eps, m, beta):
    cols = z.shape[1]
    for j in range(cols):
        t = sp.optimize.minimize(_foo, z[:, j],
                                        args=(targets[:, j], eps[:, j], m[:, j], beta))
        z[:, j] = np.reshape(t.x, (3, 1))
    return z


def _foo(z, y, eps, m, beta):
    z = np.reshape(z, (3, 1))
    count = 0
    for i in range(len(z)):
        if y[i] == 0:
            m1 = np.maximum(0, z[i])
        else:
            m1 = np.maximum(0, 1 - z[i])
        count += m1
    m2 = np.dot(z.T, eps)
    m3 = (np.linalg.norm(z - m)) ** 2
    return count + m2 + m3
