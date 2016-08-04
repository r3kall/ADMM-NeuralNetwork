import numpy as np
import numpy.matlib
import time

import auxiliaries
from logger import defineLogger, Loggers

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"

log = defineLogger(Loggers.STANDARD)


class NeuralNetwork(object):
    def __init__(self, sample_length, classes, *layers,
                 beta=1, gamma=10,
                 non_linear_func=auxiliaries.relu,
                 loss_func=auxiliaries.quadratic_cost):

        assert len(layers) > 0 and sample_length > 0 and classes > 0
        self.nl_func = non_linear_func
        self.loss_func = loss_func
        self.beta = beta
        self.gamma = gamma
        self.w = []
        self.z = []
        self.a = []

        for i in range(len(layers)):
            if i == 0:
                w = np.matlib.randn(layers[0], sample_length)
            else:
                w = np.matlib.randn(layers[i], layers[i-1])
            z = np.matlib.randn(layers[i], 1)
            self.w.append(w)
            self.z.append(z)
            self.a.append(self.nl_func(z))

        self.w.append(np.matlib.randn(classes, layers[-1]))
        self.z.append(np.matlib.randn(classes, 1))
        self.a.append(self.nl_func(self.z[-1]))
        self.dim = len(layers) + 1

    def train(self, a):
        for i in range(self.dim-1):
            if i == 0:
                self.w[i] == weight_update(self.z[i], a)
            else:
                self.w[i] = weight_update(self.z[i], self.a[i-1])
            self.a[i] = activation_update(self.w[i+1], self.z[i+1], self.a[i],
                                          self.beta, self.gamma)


def weight_update(layer_output, activation_input):
    ap_ps = np.linalg.pinv(activation_input)
    return np.dot(layer_output, ap_ps)


def _activation_inverse(next_weight, beta, gamma):
    m1 = np.dot((next_weight.T * beta), next_weight)
    m2 = np.identity(next_weight.shape[1], dtype='float64') * gamma
    return np.linalg.inv(m1 + m2)


def _activation_formulate(next_weight, next_layer_output, layer_nl_output, beta, gamma):
    m1 = np.dot((next_weight.T * beta), next_layer_output)
    m2 = gamma * layer_nl_output
    return m1 + m2


def activation_update(next_weight, next_layer_output, layer_nl_output, beta, gamma):
    m1 = _activation_inverse(next_weight, beta, gamma)
    m2 = _activation_formulate(next_weight, next_layer_output,
                               layer_nl_output, beta, gamma)
    return np.dot(m1, m2)


def argz(output, weight, activation, a, nl_fun, beta, gamma):
    m1 = gamma*(np.linalg.norm(activation - nl_fun(output))**2)
    m2 = beta*(np.linalg.norm(output - np.dot(weight, a))**2)
    return m1 + m2


def minimize(z, res, epsilon, rate, maxiter, w, act, a, nl_fun, beta, gamma):
    if res == 0:
        return z, res
    if maxiter == 0:
        return z, res

    upz = z + epsilon
    n = argz(upz, w, act, a, nl_fun, beta, gamma)
    if n < res:
        return minimize(upz, n, epsilon, rate, maxiter-1,
                        w, act, a, nl_fun, beta, gamma)

    upz = z - epsilon
    n = argz(upz, w, act, a, nl_fun, beta, gamma)
    if n < res:
        return minimize(upz, n, epsilon, rate, maxiter-1,
                        w, act, a, nl_fun, beta, gamma)
    return z, res


def getmaxz(z, res, epsilon, w, act, a, nl_fun, beta, gamma):
    n = argz(z+epsilon, w, act, a, nl_fun, beta, gamma)
    if n < res:
        return getmaxz(z+epsilon, n, epsilon, w, act, a, nl_fun, beta, gamma)
    return z, res


def getminz(z, res, epsilon, w, act, a, nl_fun, beta, gamma):
    n = argz(z-epsilon, w, act, a, nl_fun, beta, gamma)
    if n < res:
        return getmaxz(z-epsilon, n, epsilon, w, act, a, nl_fun, beta, gamma)
    return z, res


def argminz(z, res, epsilon, maxiter, w, act, a, nl_fun, beta, gamma):
    if maxiter == 0:
        return z, res
    n = argz(z + epsilon, w, act, a, nl_fun, beta, gamma)
    if n < res:
        upz, newres = getmaxz(z+epsilon, n, epsilon, w, act, a, nl_fun, beta, gamma)
        #return argminz(upz, newres, epsilon/2, maxiter-1, w, act, a, nl_fun, beta, gamma)
        return upz, newres
    n = argz(z - epsilon, w, act, a, nl_fun, beta, gamma)
    if n < res:
        upz, newres = getmaxz(z - epsilon, n, epsilon, w, act, a, nl_fun, beta, gamma)
        #return argminz(upz, newres, epsilon/2, maxiter-1, w, act, a, nl_fun, beta, gamma)
        return upz, newres
    return z, res


def main():
    w = np.matlib.randn(300, 1000)
    act = np.matlib.randn(300, 1)
    a = np.matlib.randn(1000, 1)
    nl = auxiliaries.relu
    z = np.matlib.randn(300, 1)
    res = argz(z, w, act, a, nl, 1, 10)
    print("Original res: %s" % str(res))
    st = time.time()
    for i in range(100):
        minimize(z, res, 0.1, 2, 50, w, act, a, nl, 1, 10)
    endt = time.time() - st
    print(endt)
    st = time.time()
    for i in range(100):
        argminz(z, res, 0.1, 50, w, act, a, nl, 1, 10)
    endt = time.time() - st
    print(endt)


if __name__ == "__main__":
    main()
