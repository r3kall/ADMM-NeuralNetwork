from copy import deepcopy

import numpy as np
import numpy.matlib

import time


from functions import mbhe
from neuralnetwork import NeuralNetwork, Instance
from commons import get_max_index, convert_binary_to_number


__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


def epoch(net, trn_instance, tst_instance, train_iters=1, warm_iters=0, verbose=2):
    st = time.time()
    for i in range(warm_iters):
        net.warmstart(trn_instance.samples, trn_instance.targets)
    for i in range(train_iters):
        net.train(trn_instance.samples, trn_instance.targets)
    endt = time.time() - st

    res = net.feedforward(tst_instance.samples)
    test = res.shape[1]

    c = 0
    for i in range(test):
        output = get_max_index(res[:, i])
        label = convert_binary_to_number(tst_instance.targets[:, i],
                                         tst_instance.targets.shape[0])
        if output == label:
            c += 1

    decs = 5
    approx = float(c) / float(test)
    residual = _residual(net.z[-1], net.w[-1], net.a[-1], net.beta)

    if verbose == 2:
        print("Training time: %s seconds" % np.round(endt, decimals=decs))
        loss = mbhe(res, tst_instance.targets)
        print("Mean Loss: %s" % str(np.round(loss, decimals=decs)))
        print("Residual: %s" % str(np.round(residual, decimals=decs)))
        print("Accuracy: %s/%s" % (c, test))
    if verbose != 0:
        print("Approx: %s" % str(np.round(approx, decimals=decs)))
        print("=============")
    return net, approx, residual


def _residual(z, w, a, beta):
    lr = beta * (z - (np.dot(w, a)))
    e = np.mean([np.abs(lr[k, w]) for k in range(z.shape[0]) for w in range(z.shape[1])],
                dtype=np.float64)
    return e


def adaptive_training(net, trn_instance, tst_instance,
                      accuracy=0.95, eps=0.01,
                      warm_iters=4, verbose=2):

    net, approx, residual = epoch(net, trn_instance, tst_instance,
                                  train_iters=0, warm_iters=warm_iters,
                                  verbose=verbose)
    while approx < accuracy:
        if residual > eps:
            net, approx, residual = epoch(net, trn_instance, tst_instance,
                                          train_iters=1, warm_iters=warm_iters,
                                          verbose=verbose)
        else:
            net, approx, residual = epoch(net, trn_instance, tst_instance,
                                          train_iters=1, warm_iters=0,
                                          verbose=verbose)
    return net


def binary_classification(trn_instance, tst_instance, *layers,
                          net=None, accuracy=0.95, warm_iters=4,
                          adaptive=False, verbose=2):
    if net is None:
        assert len(layers) > 0
        net = NeuralNetwork(trn_instance.samples.shape[1], trn_instance.samples.shape[0],
                            trn_instance.targets.shape[0], *layers)
    if adaptive:
        return adaptive_training(net, trn_instance, tst_instance,
                                 accuracy=accuracy, warm_iters=warm_iters, verbose=verbose)

    net, approx, residual = epoch(net, trn_instance, tst_instance,
                                  train_iters=0, warm_iters=warm_iters, verbose=verbose)
    while approx < accuracy:
        net, approx, residual = epoch(net, trn_instance, tst_instance,
                                      train_iters=1, warm_iters=0, verbose=verbose)
    return net
