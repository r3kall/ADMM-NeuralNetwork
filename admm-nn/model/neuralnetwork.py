import numpy as np

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


def hidden_output_update(weight, activation_input):

    pass


def last_output_update():
    pass


def main():
    pass


if __name__ == "__main__":
    main()
