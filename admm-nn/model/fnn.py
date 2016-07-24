#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'

import numpy as np
import numpy.matlib

class HiddenLayer(object):

    def __init__(self, n_in, n_out, n_sample, W=None, A=None, Z=None, nl_func=None,
                 beta=1, gamma=10):
        """
        Hidden layer of a MLP or FeedForward NN: units are fully-connected and have a
        custom activation function.

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type W: matrix
        :param W: weight matrix of shape(n_out, n_in)

        :type A: matrix
        :param A:

        :type Z: matrix
        :param Z:

        :type nl_func: function
        :param nl_func: Non linearity to be applied in the hidden layer
        """

        self.n_in = n_in
        self.n_out = n_out
        self.n_sample = n_sample
        self.nl_func = nl_func
        self.beta = beta
        self.gamma = gamma
        self.W = W

        if A is None:
            self.A = np.matlib.randn(self.n_out, self.n_sample)
        else:
            self.A = A

        if Z is None:
            self.Z = np.matlib.randn(self.n_out, self.n_sample)
        else:
            self.Z = Z

        np.mat(self.A, dtype='float64')
        np.mat(self.Z, dtype='float64')

    def layer_output(self):
        return self.nl_func(self.Z)

    def calc_weights(self, a_p):
        ap_ps = np.linalg.pinv(a_p)
        self.W = self.Z * ap_ps

    def calc_activation_matrix(self, beta_f, weights_f, zeta_f):
        wt = weights_f.getT() * beta_f
        w1 = wt * weights_f
        I = np.identity(weights_f.shape[1], dtype='float64') * self.gamma
        m1 = np.linalg.inv(w1 + I)

        w2 = wt * zeta_f
        h = self.layer_output() * self.gamma
        m2 = w2 + h

        self.A = m1 * m2

    def _output_matrix(self, z, a_p):
        m1 = self.gamma * np.linalg.norm()

    def calc_output_matrix(self, a_p):
        pass