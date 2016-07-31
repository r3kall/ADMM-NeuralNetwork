from abc import ABCMeta, abstractmethod
from logger import defineLogger, Loggers
from auxiliaries import check_dimensions, relu

import numpy as np
import numpy.matlib
import scipy.optimize
import auxiliaries


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

    @abstractmethod
    def layer_output(self):
        pass

    @abstractmethod
    def calc_weights(self, a_p):
        pass

    @abstractmethod
    def calc_activation_vector(self, beta_f, weights_f, zeta_f):
        pass

    @abstractmethod
    def calc_output_vector(self, a_p):
        pass


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
            self.z = np.fabs(np.matlib.randn(self.n_out, 1))
            log.debug("Output array randomly initialized")
        else:
            self.z = z
            log.debug("Output array user-defined")

        check_dimensions(self.a, self.n_out, 1)
        check_dimensions(self.a, self.n_out, 1)

    def layer_output(self):
        return self.nl_func(self.z)

    def calc_weights(self, a_p):
        ap_ps = np.linalg.pinv(a_p)
        self.w = np.dot(self.z, ap_ps)
        check_dimensions(self.w, self.n_out, self.n_in)

    def calc_activation_vector(self, beta_f, weights_f, zeta_f):
        # matrix wt is not the transpose but the adjugate
        wt = weights_f.T * beta_f
        w1 = np.dot(wt, weights_f)
        i = np.identity(weights_f.shape[1], dtype='float64') * self.gamma
        m1 = np.linalg.inv(w1 + i)

        w2 = np.dot(wt, zeta_f)
        h = self.layer_output() * self.gamma
        m2 = w2 + h

        self.a = np.dot(m1, m2)
        check_dimensions(self.a, self.n_out, 1)

    def _output_vector(self, z, mpt):
        norm1 = self.a.ravel() - relu(z)
        m1 = self.gamma * (np.linalg.norm(norm1)**2)
        norm2 = z - mpt.ravel()
        m2 = self.beta * (np.linalg.norm(norm2)**2)
        return m1 + m2

    def calc_output_vector(self, a_p):
        mpt = np.dot(self.w, a_p)
        res = scipy.optimize.minimize(self._output_vector, self.z, args=mpt)
        self.z = np.reshape(res.x, (len(res.x), 1))
        check_dimensions(self.z, self.n_out, 1)


class LastLayer(Layer):
    def __init__(self, n_in, n_out, targets, a=None, z=None, w=None, lAmbda=None):

        super().__init__(n_in, n_out, a, z, w)
        self.targets = targets

        if z is None:
            self.z = np.fabs(np.matlib.randn(self.n_out, 1))
            log.debug("Output array randomly initialized")
        else:
            self.z = z
            log.debug("Output array user-defined")

        if lAmbda is None:
            self.lAmbda = np.zeros((self.n_out, 1), dtype='float64')
            log.debug("Lambda array set to zeros")
        else:
            self.lAmbda = lAmbda
            log.debug("Lambda array user-defined")

        check_dimensions(self.z, self.n_out, 1)
        check_dimensions(self.lAmbda, self.n_out, 1)

    def layer_output(self):
        return self.nl_func(self.z)

    def calc_weights(self, a_p):
        ap_ps = np.linalg.pinv(a_p)
        self.w = np.dot(self.z, ap_ps)
        check_dimensions(self.w, self.n_out, self.n_in)

    def calc_activation_vector(self, beta_f, weights_f, zeta_f):
        raise Exception("Operation not available for %s" % LastLayer.__name__)

    def _output_vector(self, z, sp, mpt):
        # loss function
        norm = z - mpt.ravel()
        v2 = self.beta * (np.linalg.norm(norm) ** 2)
        return sp.ravel() + v2

    def calc_output_vector(self, a_p):
        sp = np.dot(self.z.T, self.lAmbda)
        mpt = np.dot(self.w, a_p)
        res = scipy.optimize.minimize(self._output_vector, self.z, args=(sp, mpt))
        self.z = np.reshape(res.x, (len(res.x), 1))
        check_dimensions(self.z, self.n_out, 1)

    def calc_lambda(self, a_p):
        wt = np.dot(self.w, a_p)
        wd = self.z - wt
        self.lAmbda += self.beta * wd


class InputLayer(Layer):
    def __init__(self, n_in, n_out, data_input, a=None, z=None, w=None):
        super().__init__(n_in, n_out, a, z, w)
        self.nl_func = auxiliaries.relu
        self.data = data_input

        self.a = self.data

    def layer_output(self):
        raise Exception("Operation not available for %s" % InputLayer.__name__)

    def calc_weights(self, a_p):
        raise Exception("Operation not available for %s" % InputLayer.__name__)

    def calc_activation_vector(self, beta_f, weights_f, zeta_f):
        raise Exception("Operation not available for %s" % InputLayer.__name__)

    def calc_output_vector(self, a_p):
        raise Exception("Operation not available for %s" % InputLayer.__name__)


def main():
    ll = LastLayer(8, 4, None)
    ap = np.log2(np.arange(8).reshape(8, 1)+0.5)
    wf = np.log2(np.arange(8).reshape(2, 4)+1.3)
    zf = np.log2(np.arange(2).reshape(2, 1)+0.7)
    ll.calc_weights(ap)
    ll.calc_output_vector(ap)
    ll.calc_lambda(ap)
    print("\nWEIGHTS")
    print(ll.w)
    print("\nOUTPUT")
    print(ll.z)
    print("\nLAMBDA")
    print(ll.lAmbda)

if __name__ == "__main__":
    main()
