import numpy as np
import scipy as sp

import scipy.optimize

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
    norm1 = activation.ravel() - nl_fun(z)
    m1 = gamma * (np.linalg.norm(norm1)**2)
    norm2 = z - mpt
    m2 = beta * (np.linalg.norm(norm2)**2)
    return m1 + m2


def minz(z, w, act, a, nl_fun, beta, gamma):
    mpt = np.squeeze(np.asarray(np.dot(w, a)))
    #org = argz(z, mpt, act, nl_fun, beta, gamma)
    #print("\nOriginal score: %s" % str(org))
    res = sp.optimize.minimize(argz, z, args=(mpt, act, nl_fun, beta, gamma))
                               #method='CG', options={'maxiter': 100, 'disp': False})
    #print("\nNew score: %s" % str(res.fun))
    return np.reshape(res.x, (len(res.x), 1))


def arglastz(z, y, loss_func, vp, mp, beta):
    m3 = beta * (np.linalg.norm(z - mp.ravel())**2)
    return loss_func(z, y.ravel()) + vp + m3


def minlastz(z, y, loss_func, zl, lAmbda, mp, beta):
    vp = np.dot(zl.T, lAmbda)[0][0]
    #print(vp)
    res = sp.optimize.minimize(arglastz, z, args=(y, loss_func, vp, mp, beta))
    return np.reshape(res.x, (len(res.x), 1))


def lambda_update(zl, mpt, beta):
    return beta * (zl - mpt)
