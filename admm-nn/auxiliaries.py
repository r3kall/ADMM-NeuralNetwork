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


def linear(x):
    return x


def mean_squared_error(prediction, label):
    assert len(prediction) == len(label)
    v = (prediction - label)**2
    s = sum(v)
    return s/len(label)


def hinge_loss(prediction, label):
    num = 0.
    dem = 0.
    if len(prediction) == len(label):
        for i in range(len(prediction)):
            prod = prediction[i] * label[i]
            num += max(0, 1 - prod)
            dem += 1
        return num/dem
    raise ValueError("Array dimensions don't match"
                     "\nPrediction length: %s"
                     "\nLabel length: %s" % (len(prediction), len(label))
                     )


def binary_classification(prediction, label):
    num = 0.
    dem = 0.
    if len(prediction) == len(label):
        for i in range(len(prediction)):
            if label[i] == 0:
                num += max(0, prediction[i])
            else:
                num += max(0, 1 - prediction[i])
            dem += 1
        return num / dem
    raise ValueError("Array dimensions don't match"
                     "\nPrediction length: %s"
                     "\nLabel length: %s" % (len(prediction), len(label))
                     )


def target_gen(classes, seed):
    t = np.full((classes, 1), 0, dtype='float64')
    t[seed] = 1
    return t


def _fill_array(array, occ, x):
    while occ > 0:
        i = random.randint(0, len(array)-1)
        c = (i+10) % len(array)
        if array[i] != x:
            array[i] = float(x)
            occ -= 1
        elif array[c] != x:
            array[c] = float(x)
            occ -= 1


def sample_gen(dim_sample, seed):
    occ = random.randint((dim_sample/4)+1, dim_sample)
    s = np.matlib.randn(dim_sample, 1)
    _fill_array(s, occ, seed)
    return s


def data_gen(dim_sample, classes, n):
    targets = []
    samples = []
    for i in range(n):
        seed = random.randint(0, 9)
        targets.append(target_gen(classes, seed))
        samples.append(sample_gen(dim_sample, seed))
    return samples, targets


def get_max_index(a):
    mx = a[0]
    index = 0
    for i in range(len(a)):
        if a[i] > mx:
            mx = a[i]
            index = i
    return mx, float(index)


def convert_binary_to_number(t):
    assert len(t) == 10
    for i in range(10):
        if t[i] == 1:
            return float(i)
    raise ValueError("Target not valid !!")