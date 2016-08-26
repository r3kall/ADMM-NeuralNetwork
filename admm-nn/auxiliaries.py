import math
import numpy as np
import random
import numpy.matlib
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


def sigmoid(x):
    return np.minimum(1, relu(x))


def linear(x):
    return x


def quadratic_cost(z, y):
    return 0.5 * ((np.abs(z - y)) ** 2)


def binary_hinge_loss(z, y):
    if y == 1:
        return np.maximum(0, 1 - z)
    return np.maximum(0, z)


def target_gen(classes, seed):
    t = np.full((classes), 0, dtype='float64')
    t[seed] = 1
    return t


def _fill_array(dim_sample, occ, x):
    s = np.full(dim_sample, 0.0001, dtype='float64')
    while occ > 0:
        i = random.randint(0, len(s)-1)
        c = (i+10) % len(s)
        if s[i] != x:
            s[i] = float(x)
            occ -= 1
        elif s[c] != x:
            s[c] = float(x)
            occ -= 1
    return s


def sample_gen(dim_sample, seed, alpha):
    occ = random.randint((dim_sample//2)+1, (dim_sample-1))
    s = _fill_array(dim_sample, occ, seed/alpha)
    return s


def data_gen(feature, classes, n):
    targets = np.zeros((classes, n))
    samples = np.zeros((feature, n))
    for i in range(n):
        seed = random.randint(0, 9)
        targets[:, i] = target_gen(classes, seed)
        samples[:, i] = sample_gen(feature, seed, 10)
    return samples, targets


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


def convert_binary_to_number(t):
    assert len(t) == 10
    for i in range(10):
        if t[i] == 1:
            return i
    raise ValueError("Target not valid !!")
