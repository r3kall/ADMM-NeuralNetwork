import numpy as np
import numpy.matlib

#from memory_profiler import profile
from model.admm import weight_update, activation_update

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


def memprofile_weight():
    z = np.matlib.randn(100, 999)
    a_in = np.matlib.randn(1000, 999)
    w = weight_update(z, a_in)


def memprofile_activation():
    wp = np.matlib.randn(100, 1000)
    zp = np.matlib.randn(100, 999)
    z = np.abs(np.matlib.randn(1000, 999))
    a = activation_update(wp, zp, z, 1., 10.)


def main():
    memprofile_weight()
    memprofile_activation()


if __name__ == "__main__":
    main()
