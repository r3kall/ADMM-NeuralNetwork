from __future__ import division
import numpy as np

cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef minimize(DTYPE_t a, DTYPE_t m, double alpha, double beta):
    if a <= 0 and m <= 0:
        return m
    cdef double sol = ((alpha * a) + (beta * m)) / (alpha + beta)
    if a >= 0 and m >= 0:
        return sol
    if m < 0 < a:
        if sol > (a**2):
            return sol
        else:
            return m
    if a < 0 < m:
        return sol


cdef minimizec(DTYPE_t a, DTYPE_t m, double alpha, double beta):
    if (alpha * a) > -(beta * m):
        return ((alpha * a) + (beta * m)) / (alpha + beta)
    return m


def argminc(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] m, double gamma, double beta):
    cdef int x = a.shape[0]
    cdef int y = a.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] z = np.zeros((x, y), dtype=DTYPE)

    for i in range(x):
        for j in range(y):
            z[i, j] = minimizec(a[i, j], m[i, j], gamma, beta)
    return z
