
import numpy as np
import numpy.matlib
import scipy.optimize
from abc import ABCMeta

from logger import defineLogger, Loggers
from auxiliaries import check_dimensions, relu, binary_classification, \
    linear, mean_squared_error

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'

log = defineLogger(Loggers.STANDARD)


class Layer(metaclass=ABCMeta):
    def __init__(self, n_in, n_out, a, z, w,
                 nl_func=None, beta=1, gamma=10):
        """
        Layer of a MLP or FeedForward NN: units are fully-connected and have a
        custom activation function.

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type a: 2D-array
        :param a: activation array of shape (n_out, 1)

        :type z: 2D-array
        :param z: output array of shape (n_out, 1)

        :type w: 2D-array
        :param w: weights array of shape(n_out, n_in)

        :type nl_func: function
        :param nl_func: Non linearity to be applied in the hidden layer

        :type beta: int
        :param beta:

        :type gamma: int
        :param gamma:
        """

        self.n_in = n_in
        self.n_out = n_out
        self.nl_func = nl_func
        self.beta = beta
        self.gamma = gamma
        self.w = w
        self.a = a
        self.z = z

    def __str__(self):
        return '[' + str(self.n_in) + ', ' + str(self.n_out) + ']'


class HiddenLayer(Layer):
    def __init__(self, n_in, n_out, a=None, z=None, w=None):
        super().__init__(n_in, n_out, a, z, w)
        self.nl_func = relu
        log.debug("Non-linear function: %s" % self.nl_func.__name__)

        if a is None:
            self.a = np.fabs(np.matlib.randn(self.n_out, 1))
            log.debug("Activation array randomly initialized")
        else:
            self.a = a
            log.debug("Activation array user-defined")

        if z is None:
            self.z = np.matlib.randn(self.n_out, 1)
            log.debug("Output array randomly initialized")
        else:
            self.z = z
            log.debug("Output array user-defined")

        if w is None:
            self.w = np.matlib.randn(self.n_out, self.n_in)
            log.debug("Weights randomly initialized")
        else:
            self.w = w
            log.debug("Weights user-defined")

        check_dimensions(self.a, self.n_out, 1)
        check_dimensions(self.z, self.n_out, 1)
        check_dimensions(self.w, self.n_out, self.n_in)

    def layer_output(self, a_p, beta_f, weights_f, zeta_f):
        self.calc_activation_array(beta_f, weights_f, zeta_f)
        self.calc_output_array(a_p)

    def calc_weights(self, a_p):
        ap_ps = np.linalg.pinv(a_p)
        self.w = np.dot(self.z, ap_ps)
        check_dimensions(self.w, self.n_out, self.n_in)

    def calc_activation_array(self, beta_f, weights_f, zeta_f):
        # matrix wt is not the transpose but the adjugate
        wt = weights_f.T * beta_f
        w1 = np.dot(wt, weights_f)
        i = np.identity(weights_f.shape[1], dtype='float64') * self.gamma
        m1 = np.linalg.inv(w1 + i)

        w2 = np.dot(wt, zeta_f)
        h = self.nl_func(self.z) * self.gamma
        m2 = w2 + h

        self.a = np.dot(m1, m2)
        check_dimensions(self.a, self.n_out, 1)

    def _output_array(self, z, mpt):
        norm1 = self.a.ravel() - self.nl_func(z)
        m1 = self.gamma * (np.linalg.norm(norm1)**2)
        norm2 = z - mpt.ravel()
        m2 = self.beta * (np.linalg.norm(norm2)**2)
        return m1 + m2

    def calc_output_array(self, a_p):
        mpt = np.squeeze(np.asarray(np.dot(self.w, a_p)))
        res = scipy.optimize.minimize(self._output_array, self.z, args=mpt)
        self.z = np.reshape(res.x, (len(res.x), 1))
        check_dimensions(self.z, self.n_out, 1)

    def train_layer(self, a_p, beta_f, weights_f, zeta_f):
        self.calc_weights(a_p)
        self.calc_activation_array(beta_f, weights_f, zeta_f)
        self.calc_output_array(a_p)


class LastLayer(Layer):
    def __init__(self, n_in, n_out, a=None, z=None, w=None, lAmbda=None):

        super().__init__(n_in, n_out, a, z, w)
        self.nl_func = linear

        if z is None:
            self.z = np.matlib.randn(self.n_out, 1)
            log.debug("%s - Output array randomly initialized" % LastLayer.__name__.upper())
        else:
            self.z = z
            log.debug("%s - Output array user-defined" % LastLayer.__name__.upper())

        if w is None:
            self.w = np.matlib.randn(self.n_out, self.n_in)
            log.debug("%s - Weights randomly initialized" % LastLayer.__name__.upper())
        else:
            self.w = w
            log.debug("%s - Weights user-defined" % LastLayer.__name__.upper())

        if lAmbda is None:
            self.lAmbda = np.zeros((self.n_out, 1), dtype='float64')
            log.debug("%s - Lambda array set to zeros" % LastLayer.__name__.upper())
        else:
            self.lAmbda = lAmbda
            log.debug("%s - Lambda array user-defined" % LastLayer.__name__.upper())

        check_dimensions(self.z, self.n_out, 1)
        check_dimensions(self.w, self.n_out, self.n_in)
        check_dimensions(self.lAmbda, self.n_out, 1)

    def layer_output(self, a_p, target):
        self.calc_output_array(a_p, target)
        self.calc_lambda(a_p)

    def calc_weights(self, a_p):
        ap_ps = np.linalg.pinv(a_p)
        self.w = np.dot(self.z, ap_ps)
        check_dimensions(self.w, self.n_out, self.n_in)

    def _output_array(self, z, sp, mpt, y):
        norm = z - mpt
        v2 = self.beta * (np.linalg.norm(norm) ** 2)
        loss = binary_classification(z, y)
        return loss + sp + v2

    def calc_output_array(self, a_p, y):
        y = np.squeeze(np.asarray(y))
        sp = (np.dot(self.z.T, self.lAmbda))[0]
        mpt = np.squeeze(np.asarray(np.dot(self.w, a_p)))
        res = scipy.optimize.minimize(self._output_array, self.z, args=(sp, mpt, y))
        self.z = np.reshape(res.x, (len(res.x), 1))
        check_dimensions(self.z, self.n_out, 1)

    def calc_lambda(self, a_p):
        wt = np.dot(self.w, a_p)
        wd = self.z - wt
        self.lAmbda += self.beta * wd
        check_dimensions(self.lAmbda, self.n_out, 1)

    def train_layer(self, a_p, target):
        self.calc_weights(a_p)
        self.calc_output_array(a_p, target)
        self.calc_lambda(a_p)


class InputLayer(Layer):
    def __init__(self, n_in, n_out, a=None, z=None, w=None):
        super().__init__(n_in, n_out, a, z, w)
        assert self.n_out == self.n_in
        self.nl_func = linear

    def layer_output(self, a_p):
        self.a = self.nl_func(a_p)


def main():
    pass

if __name__ == "__main__":
    main()
