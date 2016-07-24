
import numpy as np
import numpy.matlib
import scipy.optimize
import auxiliaries


__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


class HiddenLayer(object):

    def __init__(self, n_in, n_out, n_sample, w=None, a=None, z=None,
                 nl_func=auxiliaries.relu, beta=1, gamma=10):
        """
        Hidden layer of a MLP or FeedForward NN: units are fully-connected and have a
        custom activation function.

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type w: matrix
        :param w: weight matrix of shape(n_out, n_in)

        :type a: matrix
        :param a:

        :type z: matrix
        :param z:

        :type nl_func: function
        :param nl_func: Non linearity to be applied in the hidden layer
        """

        self.n_in = n_in
        self.n_out = n_out
        self.n_sample = n_sample
        self.nl_func = nl_func
        self.beta = beta
        self.gamma = gamma
        self.w = w

        if a is None:
            self.a = np.fabs(np.matlib.randn(self.n_out, self.n_sample))
        else:
            self.a = a

        if z is None:
            self.z = np.fabs(np.matlib.randn(self.n_out, self.n_sample))
        else:
            self.z = z

        np.mat(self.a, dtype='float64')
        np.mat(self.z, dtype='float64')

    def layer_output(self):
        return self.nl_func(self.z)

    def calc_weights(self, a_p):
        ap_ps = np.linalg.pinv(a_p)
        self.w = self.z * ap_ps

    def calc_activation_matrix(self, beta_f, weights_f, zeta_f):
        wt = weights_f.T * beta_f
        w1 = np.dot(wt, weights_f)
        I = np.identity(weights_f.shape[1], dtype='float64') * self.gamma
        m1 = np.linalg.inv(w1 + I)

        w2 = np.dot(wt, zeta_f)
        h = self.layer_output() * self.gamma
        m2 = w2 + h

        self.a = np.dot(m1, m2)

    def _output_matrix(self, z, a_p):
        norm1 = self.a.flatten() - auxiliaries.relu(z)
        m1 = self.gamma * (np.linalg.norm(norm1)**2)

        mpt = np.dot(self.w, a_p)
        norm2 = z - mpt.flatten()
        m2 = self.beta * (np.linalg.norm(norm2)**2)

        return m1 + m2

    def calc_output_matrix(self, a_p):
        res = scipy.optimize.minimize(self._output_matrix, self.z, args=a_p)
        self.z = res.x


class OutputLayer(object):
    def __init__(self, n_in, n_out, n_sample, targets, w=None, z=None,
                 lmbda=None, nl_func=None, beta=1, gamma=10):
        """
        :type
        :param n_in:

        :type
        :param n_out:

        :type
        :param n_sample:

        :type
        :param targets:

        :type
        :param w:

        :type
        :param z:

        :type
        :param lmbda:

        :type
        :param nl_func:

        :type
        :param beta:

        :type
        :param gamma:
        """

        self.n_in = n_in
        self.n_out = n_out
        self.n_sample = n_sample
        self.targets = targets
        self.nl_func = nl_func
        self.beta = beta
        self.gamma = gamma
        self.w = w

        if z is None:
            self.z = np.fabs(np.matlib.randn(self.n_out, self.n_sample))
        else:
            self.z = z
        np.mat(self.z, dtype='float64')

    def layer_output(self):
        return self.nl_func(self.z)

    def calc_weights(self, a_p):
        ap_ps = np.linalg.pinv(a_p)
        self.w = self.z * ap_ps

    def _output_matrix(self, z, a_p):
        norm1 = self.a.flatten() - auxiliaries.relu(z)
        m1 = self.gamma * (np.linalg.norm(norm1) ** 2)

        mpt = np.dot(self.w, a_p)
        norm2 = z - mpt.flatten()
        m2 = self.beta * (np.linalg.norm(norm2) ** 2)

        return m1 + m2

    def calc_output_matrix(self, a_p):
        pass






def main():
    hl1 = HiddenLayer(5, 3, 6)

    ap = np.log2(np.arange(30).reshape(5, 6)+1.8)
    mp = np.mat(ap)

    hl1.calc_weights(mp)
    hl1.calc_output_matrix(mp)
    print(hl1.z)


if __name__ == "__main__":
    main()
