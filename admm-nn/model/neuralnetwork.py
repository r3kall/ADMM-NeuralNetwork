import numpy as np
import numpy.matlib
import scipy.optimize
import time

import auxiliaries
from logger import defineLogger, Loggers
from model.admm import weight_update, activation_update, lambda_update, \
    argminz, argminlastz


__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"

log = defineLogger(Loggers.STANDARD)


class NeuralNetwork(object):
    def __init__(self, features, classes, training_space, *layers,
                 beta=1., gamma=10.,
                 non_linear_func=auxiliaries.relu,
                 loss_func=auxiliaries.binary_loss):

        assert len(layers) > 0 and features > 0 and classes > 0 and training_space > 0
        self.nl_func = non_linear_func
        self.loss_func = loss_func
        self.beta = beta
        self.gamma = gamma
        self.w = []
        self.z = []
        self.a = []
        self.lAmbda = np.zeros((classes, training_space), dtype='float64')
        #self.lAmbda = np.matlib.randn(classes, training_space)

        for i in range(len(layers)):
            if i == 0:
                w = np.matlib.randn(layers[0], features)
            else:
                w = np.matlib.randn(layers[i], layers[i-1])
            z = np.matlib.randn(layers[i], training_space)
            a = np.matlib.randn(layers[i], training_space)
            self.w.append(w)
            self.z.append(z)
            self.a.append(a)

        self.w.append(np.matlib.randn(classes, layers[-1]))
        self.z.append(np.matlib.randn(classes, training_space))
        self.a.append(self.nl_func(self.z[-1]))
        self.dim = len(layers) + 1

    def train(self, a, y):
        self._train_hidden_layers(a)

        self.w[-1] = weight_update(self.z[-1], self.a[-2])
        self.z[-1] = argminlastz(y, self.lAmbda, self.w[-1], self.a[-2], self.beta)
        self.lAmbda += lambda_update(self.z[-1], self.w[-1], self.a[-2], self.beta)

    def warmstart(self, a, y):
        self._train_hidden_layers(a)

        self.w[-1] = weight_update(self.z[-1], self.a[-2])
        self.z[-1] = argminlastz(y, self.lAmbda, self.w[-1], self.a[-2], self.beta)

    def _train_hidden_layers(self, a):
        #st = time.time()
        self.w[0] = weight_update(self.z[0], a)
        self.a[0] = activation_update(self.w[1], self.z[1], self.nl_func(self.z[0]),
                                      self.beta, self.gamma)
        #endt = time.time() - st
        self.z[0] = argminz(self.a[0], self.w[0], a, self.gamma, self.beta)
        #print("Hidden time: %s" % str(endt))

        for i in range(1, self.dim - 1):
            self.w[i] = weight_update(self.z[i], self.a[i - 1])
            self.a[i] = activation_update(self.w[i + 1], self.z[i + 1],
                                          self.nl_func(self.z[i]),
                                          self.beta, self.gamma)
            self.z[i] = argminz(self.a[i], self.w[i], self.a[i-1], self.gamma, self.beta)

    def feedforward(self, a):
        for i in range(self.dim-1):
            a = self.nl_func(np.dot(self.w[i], a))
        return np.dot(self.w[-1], a)

    def feedforward2(self, a):
        for i in range(self.dim):
            a = self.nl_func(np.dot(self.w[i], a))
        return a


def main():
    pass


if __name__ == "__main__":
    main()
