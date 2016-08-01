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
