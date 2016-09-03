import numpy as np
import numpy.matlib
import time

import auxiliaries
from functions import cost, activation, evaluation
from logger import defineLogger, Loggers, Levels
from algorithms.admm import weight_update, activation_update, \
    lambda_update, argminz, argminlastz
from neuraltools import generate_weights, generate_gaussian

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"

log = defineLogger(Loggers.STANDARD)
log.setLevel(Levels.INFO.value)


default_settings = {
    "non_linear_function"   : activation["relu"],
    "loss_function"         : cost["binary_loss"],
}


class NeuralNetwork(object):
    def __init__(self, training_space, features, classes, *layers,
                 beta=1., gamma=10., settings=default_settings):
        self.__dict__.update(settings)

        assert len(layers) > 0 and features > 0 and classes > 0 and training_space > 0
        self.parameters = (training_space, features, classes, layers)
        self.dim = len(layers) + 1

        self.beta = beta
        self.gamma = gamma

        t = (features,) + layers + (classes,)
        self.w = generate_weights(t)
        self.z = generate_gaussian(t, training_space)
        self.a = generate_gaussian(t, training_space)
        self.l = np.mat(np.zeros((classes, training_space), dtype='float64'))

    def train(self, training_data, training_targets):
        self._train_hidden_layers(training_data)
        self.w[-1] = weight_update(self.z[-1], self.a[-2])
        self.z[-1] = argminlastz(training_targets, self.l, self.w[-1], self.a[-2], self.beta)
        self.l += lambda_update(self.z[-1], self.w[-1], self.a[-2], self.beta)

    def warmstart(self, training_data, training_targets):
        self._train_hidden_layers(training_data)
        self.w[-1] = weight_update(self.z[-1], self.a[-2])
        self.z[-1] = argminlastz(training_targets, self.l, self.w[-1], self.a[-2], self.beta)

    def _train_hidden_layers(self, a):
        for i in range(self.dim - 1):
            if i == 0:
                a_in = a
            else:
                a_in = self.a[i - 1]

            self.w[i] = weight_update(self.z[i], a_in)
            self.a[i] = activation_update(self.w[i + 1], self.z[i + 1],
                                          self.non_linear_function(self.z[i]),
                                          self.beta, self.gamma)
            self.z[i] = argminz(self.a[i], self.w[i], a_in, self.gamma, self.beta)

    def feedforward(self, data):
        # This is a forward operation in the network. This is how we
        # calculate the network output from a set of input signals.
        for i in range(self.dim - 1):
            z = np.dot(self.w[i], data)
            data = self.non_linear_function(z)
        # In the last layer we don't use the activation function
        return np.dot(self.w[-1], data)

    def error(self, data, targets):
        # perform a forward operation to calculate the output signal
        out = self.feedforward(data)
        # evaluate the output signal with the loss function
        return self.loss_function(out, targets)
