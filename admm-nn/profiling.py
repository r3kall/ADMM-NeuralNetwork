import numpy as np
import numpy.matlib
import time

from model.admm import weight_update, activation_update, argminz, argminlastz, lambda_update

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


def relus(x):
    return np.maximum(0, x)


def memprofile():
    t = 0.5
    S = 11111
    indim = 1000
    outdim = 50
    hidden1 = 500
    hidden2 = 100
    a0 = np.abs(np.matlib.randn(indim, S))

    z1 = np.matlib.randn(hidden1, S)
    time.sleep(t)
    w1 = weight_update(z1, a0)
    time.sleep(t)

    nw1 = np.mat(np.zeros((hidden2, hidden1), dtype='float64'))
    time.sleep(t)
    a1 = activation_update(nw1, np.matlib.randn(hidden2, S), relus(z1), 1., 10.)
    time.sleep(t)
    z1 = argminz(a1, w1, a0, 10., 1.)
    time.sleep(t)

    z2 = np.matlib.randn(hidden2, S)
    time.sleep(t)
    w2 = weight_update(z2, a1)
    time.sleep(t)

    nw2 = np.mat(np.zeros((outdim, hidden2), dtype='float64'))
    time.sleep(t)
    a2 = activation_update(nw2, np.matlib.randn(outdim, S), relus(z2), 1., 10.)
    time.sleep(t)
    z2 = argminz(a2, w2, a1, 10., 1.)
    time.sleep(t)

    z3 = np.matlib.randn(outdim, S)
    time.sleep(t)
    w3 = weight_update(z3, a2)
    time.sleep(t)
    #z3 = argminlastz()
    l = lambda_update(z3, w3, a2, 1.)
    time.sleep(t)


def main():
    memprofile()


if __name__ == "__main__":
    main()
