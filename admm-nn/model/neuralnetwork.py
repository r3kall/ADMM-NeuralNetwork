import numpy as np
import numpy.matlib
import time

import auxiliaries
from logger import defineLogger, Loggers, Levels
from model.admm import weight_update, activation_update, \
    lambda_update, argminz, argminlastz


__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"

log = defineLogger(Loggers.STANDARD)
log.setLevel(Levels.INFO.value)


class NeuralNetwork(object):
    def __init__(self, features, classes, training_space, *layers,
                 beta=1., gamma=10.,
                 non_linear_func=auxiliaries.relu,
                 loss_func=auxiliaries.binary_loss):

        start = time.time()
        assert len(layers) > 0 and features > 0 and classes > 0 and training_space > 0
        self.nl_func = non_linear_func
        self.loss_func = loss_func
        self.beta = beta
        self.gamma = gamma
        self.w = []
        self.z = []
        self.a = []
        self.lAmbda = np.mat(np.zeros((classes, training_space), dtype='float64'))

        for i in range(len(layers)):
            if i == 0:
                w = np.mat(np.zeros((layers[0], features), dtype='float64'))
            else:
                w = np.mat(np.zeros((layers[i], layers[i-1]), dtype='float64'))
            z = np.matlib.randn(layers[i], training_space)
            a = np.matlib.randn(layers[i], training_space)
            self.w.append(w)
            self.z.append(z)
            self.a.append(a)
        self.w.append(np.mat(np.zeros((classes, layers[-1]), dtype='float64')))
        self.z.append(np.matlib.randn(classes, training_space))
        self.a.append(self.nl_func(self.z[-1]))
        self.dim = len(layers) + 1

        for i in range(self.dim):
            auxiliaries.check_consistency(self.w[i])
            auxiliaries.check_consistency(self.a[i])
            auxiliaries.check_consistency(self.z[i])
            auxiliaries.check_consistency(self.lAmbda)
        endt = time.time() - start
        log.debug("Neural Network - Creation time: %s seconds" %
                  str(np.round(endt, decimals=6)))

    def train(self, a, y):
        self._train_hidden_layers(a)
        start = time.time()
        self.w[-1] = weight_update(self.z[-1], self.a[-2])
        self.z[-1] = argminlastz(y, self.lAmbda, self.w[-1], self.a[-2], self.beta)
        self.lAmbda += lambda_update(self.z[-1], self.w[-1], self.a[-2], self.beta)
        endt = time.time() - start
        log.debug("Neural Network - Last layer Training time: %s seconds" %
                  str(np.round(endt, decimals=6)))

    def warmstart(self, a, y):
        self._train_hidden_layers(a)
        start = time.time()
        self.w[-1] = weight_update(self.z[-1], self.a[-2])
        self.z[-1] = argminlastz(y, self.lAmbda, self.w[-1], self.a[-2], self.beta)
        endt = time.time() - start
        log.debug("Neural Network - Last layer Warmstart time: %s seconds" %
                  str(np.round(endt, decimals=6)))

    def _train_hidden_layers(self, a):
        start = time.time()
        self.w[0] = weight_update(self.z[0], a)
        self.a[0] = activation_update(self.w[1], self.z[1], self.nl_func(self.z[0]),
                                      self.beta, self.gamma)
        self.z[0] = argminz(self.a[0], self.w[0], a, self.gamma, self.beta)

        for i in range(1, self.dim - 1):
            self.w[i] = weight_update(self.z[i], self.a[i - 1])
            self.a[i] = activation_update(self.w[i + 1], self.z[i + 1],
                                          self.nl_func(self.z[i]),
                                          self.beta, self.gamma)
            self.z[i] = argminz(self.a[i], self.w[i], self.a[i-1], self.gamma, self.beta)
        endt = time.time() - start
        log.debug("Neural Network - Hidden layers Training time: %s seconds" %
                  str(np.round(endt, decimals=6)))

    def feedforward(self, a):
        for i in range(self.dim-1):
            z = np.dot(self.w[i], a)
            a = self.nl_func(z)
        return np.dot(self.w[-1], a)

    def feedforward2(self, a):
        for i in range(self.dim):
            a = self.nl_func(np.dot(self.w[i], a))
        return a
