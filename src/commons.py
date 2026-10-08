import numpy as np

from .logger import defineLogger, Loggers

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
    if n <= 0 or m <= 0:
        raise ValueError("dimensions must be positive")
    check_consistency(a)
    if a.shape[0] != n or a.shape[1] != m:
        raise ValueError("Invalid dimensions of the array")
    log.debug("Function '%s' validates the array" % check_dimensions.__name__)


def get_max_index(a):
    # Get the index with maximum value
    mx = a[0]
    index = 0
    for i in range(len(a)):
        if a[i] > mx:
            mx = a[i]
            index = i
    return index


def convert_binary_to_number(t, dim):
    if len(t) != dim:
        raise ValueError("target length does not match classes")
    for i in range(dim):
        if t[i] == 1:
            return i
    raise ValueError("Target not valid !!")


def get_percentage(percentage, n):
    if not np.isfinite(percentage) or not 0 <= percentage <= 100:
        raise ValueError("percentage must be finite and between 0 and 100")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("sample count must be a nonnegative integer")
    return int(np.floor(n * percentage / 100))
