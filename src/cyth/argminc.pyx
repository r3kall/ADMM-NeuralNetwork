from __future__ import division
import numpy as np

cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef minimize(DTYPE_t a, DTYPE_t m, double alpha, double beta):
    cdef DTYPE_t sol = ((alpha * a) + (beta * m)) / (alpha + beta)
    if m >= 0:
        if sol > 0:
            return sol
        return 0
    else:
        if a <= 0:
            return m
        res1 = (alpha * ((a - (np.maximum(0, sol))) ** 2)) + (beta * ((sol - m) ** 2))
        res2 = alpha * (a ** 2)
        if res1 <= res2:
            return sol
        return m



def argminc(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] m, double gamma, double beta):
    cdef int x = a.shape[0]
    cdef int y = a.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] z = np.zeros((x, y), dtype=DTYPE)

    for i in range(x):
        for j in range(y):
            z[i, j] = minimize(a[i, j], m[i, j], gamma, beta)
    return z
