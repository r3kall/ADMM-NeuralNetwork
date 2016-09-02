import math
import numpy as np
from logger import defineLogger, Loggers

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'

log = defineLogger(Loggers.STANDARD)


def check_consistency(a):
    if a is None:
        raise TypeError("Invalid type: None")
    if not isinstance(a, np.ndarray):
        raise TypeError("Invalid type: %s" % type(a).__name__)
    if len(a.shape) != 2:
        raise ValueError("Invalid shape of the array.\n"
                         "Actual dimensions: %s" % len(a.shape))
    log.debug("Function '%s' validates the array" % check_consistency.__name__)


def check_dimensions(a, n, m):
    assert n > 0, m > 0
    check_consistency(a)
    if a.shape[0] != n or a.shape[1] != m:
        raise ValueError("Invalid dimensions of the array")
    log.debug("Function '%s' validates the array" % check_dimensions.__name__)


def relu(x):
    return np.maximum(0, x)


def binary_loss(z, y):
    if y == 1:
        return np.maximum(0, 1 - z)
    return np.maximum(0, z)


def binary_loss_sum(z, y):
    c = 0
    for j in range(z.shape[1]):
        for i in range(z.shape[0]):
            c += binary_loss(z[i, j], y[i, j])
    return c / (z.shape[0] * z.shape[1])


def get_max_index(a):
    mx = a[0]
    index = 0
    for i in range(len(a)):
        if a[i] > mx:
            mx = a[i]
            index = i
    return index


def get_percentage(percentage, n):
    assert 0 <= percentage <= 100
    if percentage == 0:
        return 0
    if percentage == 100:
        return n
    return math.floor((n/100)*percentage)


def convert_binary_to_number(t, dim):
    assert len(t) == dim
    for i in range(dim):
        if t[i] == 1:
            return i
    raise ValueError("Target not valid !!")
