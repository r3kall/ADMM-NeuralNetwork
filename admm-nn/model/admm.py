import numpy as np
import scipy as sp

import scipy.optimize
import time

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def weight_update(layer_output, activation_input):
    ap_ps = np.linalg.pinv(activation_input)
    return np.dot(layer_output, ap_ps)


def _activation_inverse(next_weight, beta, gamma):
    m1 = np.dot((next_weight.H * beta), next_weight)
    m2 = np.identity(next_weight.shape[1], dtype='float64') * gamma
    return np.linalg.inv(m1 + m2)


def _activation_formulate(next_weight, next_layer_output, layer_nl_output, beta, gamma):
    m1 = np.dot((next_weight.H * beta), next_layer_output)
    m2 = gamma * layer_nl_output
    return m1 + m2


def activation_update(next_weight, next_layer_output, layer_nl_output, beta, gamma):
    m1 = _activation_inverse(next_weight, beta, gamma)
    m2 = _activation_formulate(next_weight, next_layer_output,
                               layer_nl_output, beta, gamma)
    return np.dot(m1, m2)


def argz(z, mpt, activation, nl_fun, beta, gamma):
    #st = time.time()
    norm1 = activation.ravel() - nl_fun(z)
    m1 = gamma * (np.linalg.norm(norm1)**2)
    norm2 = z - mpt.ravel()
    m2 = beta * (np.linalg.norm(norm2)**2)
    #endt = time.time() - st
    #print("Argz time: %s" % str(round(endt, 4)))
    return m1 + m2


def minz(z, w, act, a, nl_fun, beta, gamma):
    mpt = np.dot(w, a)
    for j in range(z.shape[1]):
        #org = argz(z[:, j], mpt[:, j], act[:, j], nl_fun, beta, gamma)
        #print("\nOriginal score: %s" % str(org))
        st = time.time()
        res = sp.optimize.minimize(argz, z[:, j], args=(mpt[:, j], act[:, j], nl_fun, beta, gamma),
                                   method='L-BFGS-B', options={'maxiter': 1000, 'disp': False})
        endt = time.time() - st
        #print("Argz time: %s" % str(round(endt, 4)))
        #z[:, j] = np.reshape(res.x, (z.shape[0], 1))
        #print("New score: %s" % str(res.fun))
    return z


def arglastz(z, y, loss_func, vp, mp, beta):
    m3 = beta * (np.linalg.norm(z - mp.ravel()[0])**2)
    return loss_func(z, y.ravel()) + vp + m3


def minlastz(z, y, loss_func, zl, lAmbda, mp, beta):
    vp = np.inner(zl.ravel(), lAmbda.ravel())
    res = sp.optimize.minimize(arglastz, z, args=(y, loss_func, vp, mp, beta))
    #print(res.fun)
    return np.reshape(res.x, (z.shape[0], z.shape[1]))


def lambda_update(zl, mpt, beta):
    return beta * (zl - mpt)
