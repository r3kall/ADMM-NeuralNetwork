from __future__ import division
import numpy as np

cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


cdef minbhe(np.uint8_t y, DTYPE_t eps, DTYPE_t m, double beta):
    cdef DTYPE_t w = m - (eps / (2 * beta))
    cdef DTYPE_t t = 1 / (2 * beta)
    if y == 0:
        if w <= 0:
            return w
        if w <= t:
            return 0.0
        return w - t
    if w >= 1:
        return w
    if w >= 1 - t:
        return 1.0
    return w + t


def binarymin(np.ndarray[np.uint8_t, ndim=2] targets,
              np.ndarray[DTYPE_t, ndim=2] eps,
              np.ndarray[DTYPE_t, ndim=2] m,
              double beta):

    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if (targets.shape[0] != eps.shape[0] or targets.shape[1] != eps.shape[1]
            or targets.shape[0] != m.shape[0] or targets.shape[1] != m.shape[1]):
        raise ValueError("targets, eps and m must have equal shapes")
    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("targets must contain only 0 and 1")
    if not np.isfinite(eps).all() or not np.isfinite(m).all():
        raise ValueError("eps and m must be finite")
    cdef int x = targets.shape[0]
    cdef int y = targets.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] z = np.zeros((x, y), dtype=DTYPE)

    for i in range(x):
        for j in range(y):
            z[i, j] = minbhe(targets[i, j], eps[i, j], m[i, j], beta)
    return z
