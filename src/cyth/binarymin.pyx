from __future__ import division
import numpy as np

cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef minbhe(np.uint8_t y, DTYPE_t eps, DTYPE_t m, double beta):
    if y == 0:
        if m > ((1 + eps) / (2 * beta)):
            return m - ((1 + eps) / (2 * beta))
        else:
            return m - (eps / (2 * beta))
    else:
        sol = m + ((1 - eps) / (2 * beta))
        if sol < 1 and sol != 0:
            return sol
        else:
            return m - (eps / (2 * beta))


def binarymin(np.ndarray[np.uint8_t, ndim=2] targets,
              np.ndarray[DTYPE_t, ndim=2] eps,
              np.ndarray[DTYPE_t, ndim=2] m,
              double beta):

    cdef int x = targets.shape[0]
    cdef int y = targets.shape[1]
    cdef np.ndarray z = np.zeros((x, y), dtype=DTYPE)

    for i in range(x):
        for j in range(y):
            z[i, j] = minbhe(targets[i, j], eps[i, j], m[i, j], beta)
    return z
