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


npa = np.array
def softmax(w, t = 1.0):
    e = np.exp(npa(w) / t)
    dist = e / np.sum(e)
    return dist


def relu(x):
    return np.maximum(0, x)


def binary_loss(z, y):
    if y == 1:
        return np.maximum(0, 1 - z)
    return np.maximum(0, z)


def binary_loss_sum(z, y):
    c = 0
    for j in range(z.shape[1]):
        t = 0
        for i in range(z.shape[0]):
            t += binary_loss(z[i, j], y[i, j])
        c += t / z.shape[0]
    return c / z.shape[1]


def target_gen(classes, seed):
    t = np.full(classes, 0, dtype='float64')
    t[seed] = 1
    return t


def _fill_array(dim_sample, occ, x):
    s = np.full(dim_sample, 0.001, dtype='float64')
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
    #s = np.full(dim_sample, seed/alpha, dtype='float64')
    return s


def data_gen(feature, classes, n):
    targets = np.zeros((classes, n))
    samples = np.zeros((feature, n))
    for i in range(n):
        seed = random.randint(0, 9)
        targets[:, i] = target_gen(classes, seed)
        samples[:, i] = sample_gen(feature, seed, 10)
    return samples, targets


def triple_data_gen(feature, classes, n):
    assert classes == 4
    targets = np.zeros((classes, n))
    samples = np.zeros((feature, n))
    for i in range(n):
        seed = random.randint(0, 3)
        targets[:, i] = target_gen(classes, seed)
        samples[:, i] = sample_gen(feature, seed, 1)
    return samples, targets


def convert_triple_to_number(t):
    assert len(t) == 4
    for i in range(4):
        if t[i] == 1:
            return i
    raise ValueError("Target not valid !!")


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
