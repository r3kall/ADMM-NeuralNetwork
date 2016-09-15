from __future__ import division
import numpy as np

cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef minbhe(np.uint8_t y, DTYPE_t eps, DTYPE_t m, double beta):
    cdef DTYPE_t w = m - (eps / (2 * beta))
    cdef DTYPE_t s0 = m - ((1 + eps) / (2 * beta))
    cdef DTYPE_t s1 = m + ((1 - eps) / (2 * beta))

    if y == 0:
        if s0 + w >= 0:
            return s0
        return w
    else:
        if s1 + w <= 2:
            return s1
        return w


def binarymin(np.ndarray[np.uint8_t, ndim=2] targets,
              np.ndarray[DTYPE_t, ndim=2] eps,
              np.ndarray[DTYPE_t, ndim=2] m,
              double beta):

    cdef int x = targets.shape[0]
    cdef int y = targets.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] z = np.zeros((x, y), dtype=DTYPE)

    for i in range(x):
        for j in range(y):
            z[i, j] = minbhe(targets[i, j], eps[i, j], m[i, j], beta)
    return z
