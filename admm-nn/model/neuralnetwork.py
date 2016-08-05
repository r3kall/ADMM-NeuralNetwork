import numpy as np
import numpy.matlib
import scipy as sp
import scipy.optimize
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
        self.lAmbda = np.zeros((classes, 1), dtype='float64')

        for i in range(len(layers)):
            if i == 0:
                w = np.matlib.randn(layers[0], sample_length)
            else:
                w = np.matlib.randn(layers[i], layers[i-1])
            z = np.matlib.randn(layers[i], 1)
            a = np.matlib.randn(layers[i], 1)
            self.w.append(w)
            self.z.append(z)
            self.a.append(a)

        self.w.append(np.matlib.randn(classes, layers[-1]))
        self.z.append(np.matlib.randn(classes, 1))
        self.a.append(self.nl_func(self.z[-1]))
        self.dim = len(layers) + 1

    def train(self, a, y):
        self.w[0] = weight_update(self.z[0], a)
        self.a[0] = activation_update(self.w[1], self.z[1], self.a[0],
                                      self.beta, self.gamma)
        #st = time.time()
        self.z[0] = minz(self.z[0], self.w[0], self.a[0], a, auxiliaries.relu, 1, 10)
        #endt = time.time() - st
        #print("TIME: %s\n" % str(endt))

        for i in range(1, self.dim-1):
            self.w[i] = weight_update(self.z[i], self.a[i-1])
            self.a[i] = activation_update(self.w[i+1], self.z[i+1], self.a[i],
                                          self.beta, self.gamma)
            self.z[i] = minz(self.z[i], self.w[i], self.a[i], self.a[i - 1],
                             self.nl_func, self.beta, self.gamma)

        self.w[-1] == weight_update(self.z[-1], self.z[-2])
        mp = np.dot(self.w[-1], self.a[-2])
        self.z[-1] == minlastz(self.z[-1], y, self.loss_func, self.z[-1],
                               self.lAmbda, mp, self.beta)
        self.lAmbda += lambda_update(self.z[-1], mp, self.beta)


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
    res = sp.optimize.minimize(argz, z, args=(mpt, act, nl_fun, beta, gamma),
                               method='CG', options={'maxiter': 100})
    #print("\nNew score: %s" % str(res.fun))
    return np.reshape(res.x, (len(res.x), 1))


def arglastz(z, y, loss_func, vp, mp, beta):
    m3 = beta * (np.linalg.norm(z - mp.ravel())**2)
    return loss_func(z, y.ravel()) + vp + m3


def minlastz(z, y, loss_func, zl, lAmbda, mp, beta):
    vp = np.dot(zl.T, lAmbda)
    res = sp.optimize.minimize(arglastz, z, args=(y, loss_func, vp, mp, beta))
    return np.reshape(res.x, (len(res.x), 1))


def lambda_update(zl, mpt, beta):
    return beta * (zl - mpt)


def main():
    a = np.matlib.randn(500, 1)
    y = np.fabs(np.matlib.randn(10, 1))
    nn = NeuralNetwork(500, 10, 100)
    nn.train(a, y)


if __name__ == "__main__":
    main()
