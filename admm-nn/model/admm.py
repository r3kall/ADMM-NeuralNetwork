import numpy as np


__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


def weight_update(layer_output, activation_input):
    ap_ps = np.linalg.pinv(activation_input)
    return np.dot(layer_output, ap_ps)


def _activation_inverse(next_weight, beta, gamma):
    m1 = beta * (np.dot(next_weight.H, next_weight))
    m2 = (np.identity(next_weight.shape[1], dtype='float64')) * gamma
    return np.linalg.inv(m1 + m2)


def _activation_formulate(next_weight, next_layer_output, layer_nl_output, beta, gamma):
    m1 = beta * (np.dot(next_weight.H, next_layer_output))
    m2 = gamma * layer_nl_output
    return m1 + m2


def activation_update(next_weight, next_layer_output, layer_nl_output, beta, gamma):
    m1 = _activation_inverse(next_weight, beta, gamma)
    m2 = _activation_formulate(next_weight, next_layer_output,
                               layer_nl_output, beta, gamma)
    return np.dot(m1, m2)


def _minimize(a, m, alpha, beta):
    if a <= 0 and m <= 0:
        return m
    sol = ((alpha * a) + (beta * m)) / (alpha + beta)
    if a >= 0 and m >= 0:
        return sol
    if m < 0 < a:
        if sol / (a ** 2) > 1:
            return sol
        else:
            return m
    if a < 0 < m:
        return sol


def argminz(a, w, a_in, gamma, beta):
    m = np.dot(w, a_in)
    x = a.shape[0]
    y = a.shape[1]
    z = np.zeros((x, y), dtype='float64')
    for i in range(x):
        for j in range(y):
            z[i, j] = _minimize(a[i, j], m[i, j], gamma, beta)
    return z


def _minimizelast(y, eps, m, beta):
    if y == 0:
        if m > (eps + 1) / (2 * beta):
            return m - ((eps + 1) / (2 * beta))
        else:
            return m - (eps / (2 * beta))
    else:
        if m < 1 + ((eps - 1) / (2 * beta)):
            return m - ((eps - 1) / (2 * beta))
        else:
            return m - (eps / (2 * beta))


def argminlastz(targets, eps, w, a_in, beta):
    m = np.dot(w, a_in)
    x = targets.shape[0]
    y = targets.shape[1]
    z = np.zeros((x, y), dtype='float64')
    for i in range(x):
        for j in range(y):
            z[i, j] = _minimizelast(targets[i, j], eps[i, j], m[i, j], beta)
    return z


def lambda_update(zl, w, a_in, beta):
    mpt = np.dot(w, a_in)
    return beta * (zl - mpt)
