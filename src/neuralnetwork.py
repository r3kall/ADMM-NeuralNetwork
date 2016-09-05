import numpy as np

from functions import cost, activation, evaluation
from logger import defineLogger, Loggers, Levels
from algorithms.admm import weight_update, activation_update, \
    lambda_update, argminz, argminlastz
from neuraltools import generate_weights, generate_gaussian

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"

log = defineLogger(Loggers.STANDARD)
log.setLevel(Levels.INFO.value)


default_settings = {
    "activation_function"   : activation["relu"],
    "cost_function"         : cost["binary_loss"],
    "error_function"        : evaluation["classification_error"]
}


class NeuralNetwork(object):
    # Neural Network Model
    def __init__(self, training_space, features, classes, *layers,
                 beta=1.0, gamma=10.0, settings=default_settings):
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
    # end

    def train(self, training_data, training_targets):
        self._train_hidden_layers(training_data)
        self.w[-1] = weight_update(self.z[-1], self.a[-2])
        self.z[-1] = argminlastz(training_targets, self.l, self.w[-1], self.a[-2], self.beta)
        self.l += lambda_update(self.z[-1], self.w[-1], self.a[-2], self.beta)
    # end

    def warmstart(self, training_data, training_targets):
        # Train the net without the Lagrangian update
        self._train_hidden_layers(training_data)
        self.w[-1] = weight_update(self.z[-1], self.a[-2])
        self.z[-1] = argminlastz(training_targets, self.l, self.w[-1], self.a[-2], self.beta)
    # end

    def _train_hidden_layers(self, a):
        for i in range(self.dim - 1):
            if i == 0:
                a_in = a
            else:
                a_in = self.a[i - 1]

            self.w[i] = weight_update(self.z[i], a_in)
            self.a[i] = activation_update(self.w[i + 1], self.z[i + 1],
                                          self.activation_function(self.z[i]),
                                          self.beta, self.gamma)
            self.z[i] = argminz(self.a[i], self.w[i], a_in, self.gamma, self.beta)
    # end

    def feedforward(self, data):
        # This is a forward operation in the network. This is how we
        # calculate the network output from a set of input signals.
        for i in range(self.dim - 1):
            z = np.dot(self.w[i], data)
            data = self.activation_function(z)
        # In the last layer we don't use the activation function
        return np.dot(self.w[-1], data)
    # end

    def error(self, data, targets):
        # perform a forward operation to calculate the output signal
        out = self.feedforward(data)
        # evaluate the output signal with the evaluation function
        return self.error_function(out, targets)
    # end
# end class NeuralNetwork


class Instance:
    # This is a simple encapsulation of a `input signal : output signal`
    # pair in our training set.
    def __init__(self, samples, targets):
        self.samples = np.mat(samples)
        self.targets = np.mat(targets)
# end class Instance