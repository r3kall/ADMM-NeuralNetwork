import numpy as np

from algorithms.admm import weight_update, activation_update, argminz, lambda_update
from neuraltools import generate_weights, generate_outputs, generate_activations

from logger import defineLogger, Loggers, Levels
log = defineLogger(Loggers.STANDARD)
log.setLevel(Levels.INFO.value)

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


class NeuralNetwork(object):
    # Neural Network Model
    def __init__(self, training_space, features, classes, *layers,
                 beta=1., gamma=10., code='binary'):

        assert len(layers) > 0 and features > 0 and classes > 0 and training_space > 0
        self.parameters = (training_space, features, classes, layers)
        self.dim = len(layers) + 1

        self.beta = beta
        self.gamma = gamma

        self.argminlastz, self.activation_function, self.error = setalg(code)

        t = (features,) + layers + (classes,)
        self.w = generate_weights(t)
        self.z = generate_outputs(t, training_space)
        self.a = generate_activations(t, training_space)
        self.l = np.mat(np.zeros((classes, training_space), dtype='float64'))
    # end

    def train(self, training_data, training_targets):
        self._train_hidden_layers(training_data)
        self.w[-1] = weight_update(self.z[-1], self.a[-1])
        self.z[-1] = self.argminlastz(training_targets, self.l, self.w[-1], self.a[-1], self.beta)
        self.l += lambda_update(self.z[-1], self.w[-1], self.a[-1], self.beta)
    # end

    def warmstart(self, training_data, training_targets):
        # Train the net without the Lagrangian update
        self._train_hidden_layers(training_data)
        self.w[-1] = weight_update(self.z[-1], self.a[-1])
        self.z[-1] = self.argminlastz(training_targets, self.l, self.w[-1], self.a[-1], self.beta)
    # end

    def _train_hidden_layers(self, a):
        self.w[0] = weight_update(self.z[0], a)
        self.a[0] = activation_update(self.w[1], self.z[1],
                                      self.activation_function(self.z[0]),
                                      self.beta, self.gamma)
        self.z[0] = argminz(self.a[0], self.w[0], a, self.gamma, self.beta)

        for i in range(1, self.dim - 1):
            self.w[i] = weight_update(self.z[i], self.a[i - 1])
            self.a[i] = activation_update(self.w[i + 1], self.z[i + 1],
                                          self.activation_function(self.z[i]),
                                          self.beta, self.gamma)
            self.z[i] = argminz(self.a[i], self.w[i], self.a[i - 1], self.gamma, self.beta)
    # end

    def feedforward(self, data):
        # This is a forward operation in the network. This is how we
        # calculate the network output from a set of input signals.
        for i in range(self.dim - 1):
            data = self.activation_function(np.dot(self.w[i], data))
        # In the last layer we don't use the activation function
        return np.dot(self.w[-1], data)
    # end

    def error(self, data, targets):
        # perform a forward operation to calculate the output signal
        out = self.feedforward(data)
        # evaluate the output signal with the evaluation function
        return self.error(out, targets)
    # end
# end class NeuralNetwork


def setalg(code):
    if code == 'binary':
        from algorithms.hingebinary import argminlastz
        from functions import relu, mbhe
        return argminlastz, relu, mbhe
    else:
        return None, None
# end


class Instance(object):
    # This is a simple encapsulation of a `input signal : output signal`
    # pair in our training set.
    def __init__(self, samples, targets, intype=np.float64, outtype=np.uint8):
        from commons import check_consistency
        self.samples = np.mat(samples, dtype=intype)
        self.targets = np.mat(targets, dtype=outtype)
        check_consistency(self.samples)
        check_consistency(self.targets)
# end class Instance
