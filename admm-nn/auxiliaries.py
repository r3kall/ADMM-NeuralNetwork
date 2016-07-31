import numpy as np

from logger import defineLogger, Loggers

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'

log = defineLogger(Loggers.STANDARD)


def relu(x):
    return np.maximum(0, x)


def print_limited_matrix(x, n, m):
    # da migliorare, meglio 'prin_matrix_info', in cui usare 'check_type'
    # if check_type ok, se matrice get_matrix_info, oppure get_vector_info
    # if n <= 1 or n > x.dim or None raise
    for i in range(n):
        for j in range(m):
            print(x[i][j])


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
