import pytest

import numpy as np
import scipy as sp
import numpy.matlib
import scipy.optimize
import time

import auxiliaries

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def argz(z, a, mp, nl_func, beta=1, gamma=10):
    norm1 = a - nl_func(z)
    norm2 = z - mp
    m1 = gamma * (np.linalg.norm(norm1, ord=2) ** 2)
    m2 = beta * (np.linalg.norm(norm2, ord=2) ** 2)
    return m1 + m2


def minz(zv, zm, j, n, a, mp, nl_func, beta=1, gamma=10):
    zv = np.reshape(zv, (n, 1))
    zm[:, j] = zv
    return argz(zm, a, mp, nl_func)


def test_1():
    """
    Given a matrix NxS (samples) named Z, we want to minimize the function argz,
    but entry-wise, so we want to minimize each Z entry (Nx1) indipendently.
    First Test:
    Q1) the given column is minimized ? the others columns changes without permission?
    """
    print()
    outdim = 10
    indim = 1000
    n = 1024

    z = np.matlib.randn(outdim, n)
    a = np.matlib.randn(outdim, n)
    w = np.matlib.randn(outdim, indim)
    j = np.matlib.randn(indim, n)
    mp = np.dot(w, j)

    zv = z[:, 0]
    print("original matrix check")
    print(z[:, 2])
    print("Initial x")
    print(zv.flatten())
    res = sp.optimize.minimize(minz, zv, args=(z, 0, outdim, a, mp, auxiliaries.relu))
    res = res.x
    print("Minimized x")
    print(res.flatten())

    print("Original matrix check")
    print(z[:, 2])


def test_2():
    """
    When outdim increase, performance seems to slow so much
    Q1) what's the cause of the slowness ?
    Q2) Check if it is ARGZ
    Q3) Check if it is the minimize method
    """
    print()
    outdim = 512
    indim = 1024
    n = 4096

    z = np.matlib.randn(outdim, n)
    a = np.matlib.randn(outdim, n)
    w = np.matlib.randn(outdim, indim)
    j = np.matlib.randn(indim, n)
    mp = np.dot(w, j)

    st = time.time()
    res = argz(z, a, mp, auxiliaries.relu)
    endt = time.time() - st
    print("Time Argz: %s" % str(round(endt, 4)))

    """
    ANSWER: well, seems that the cause of the slowness could be the number of iteration
    in the minimize method. This is cause of the Argz function that for large numbers of
    outdim and n can use several seconds (for one minimize iteration).
    Next test: use the minimize function changing the method and other parameters
    """


def test_3():
    print()
    outdim = 256
    indim = 768
    n = 128

    z = np.matlib.randn(outdim, n)
    a = np.matlib.randn(outdim, n)
    w = np.matlib.randn(outdim, indim)
    j = np.matlib.randn(indim, n)
    mp = np.dot(w, j)

    score = argz(z, a, mp, auxiliaries.relu)
    print("Original score: %s\n" % str(score))

    for j in range(n):
        zv = z[:, j]
        st = time.time()

        res = sp.optimize.minimize(minz, zv, args=(z, j, outdim, a, mp, auxiliaries.relu),
                                   method='L-BFGS-B',
                                   options={'maxcor':10, 'maxiter':100, 'maxfun':100})

        endt = time.time() - st
        z[:, j] = np.reshape(res.x, (outdim, 1))
        print(round(endt, 4))

    finalscore = argz(z, a, mp, auxiliaries.relu)
    print("\nFinal score: %s" % str(finalscore))


def arglastz(z, y, loss_func, vp, mp, beta):
    norm = z - mp
    m3 = beta * (np.linalg.norm(norm, ord=2) ** 2)
    loss = loss_func(z, y)
    return loss + vp + m3


def minlastz(zv, zm, pos, dim, y, loss_func, vp, mp, beta):
    zv = np.reshape(zv, (dim, 1))
    zm[:, pos] = zv
    return arglastz(zm, y, loss_func, vp, mp, beta)


def argminlastz(z, y, lAmbda, mp, loss_func, beta):
    vp = 0
    outdim = z.shape[0]
    for j in range(z.shape[1]):
        res = sp.optimize.minimize(minlastz, z[:, j],
                                   args=(z, j, outdim, y, loss_func, vp, mp, beta),
                                   method='L-BFGS-B')
        z[:, j] = np.reshape(res.x, (outdim, 1))
    return z


def test_4():
    print()
    outdim = 16
    indim = 512
    n = 1

    z = np.matlib.randn(outdim, n)
    y = np.matlib.randn(outdim, n)
    w = np.matlib.randn(outdim, indim)
    j = np.matlib.randn(indim, n)
    loss = auxiliaries.quadratic_cost
    mp = np.dot(w, j)

    print("ORIGINAL SCORE")
    score = arglastz(z, y, loss, 0, mp, 1)
    print(score)
    st = time.time()
    z = argminlastz(z, y, 0, mp, loss, 1)
    endt = time.time() - st
    print(endt)
    print("\nFINAL SCORE")
    score = arglastz(z, y, loss, 0, mp, 1)
    print(score)