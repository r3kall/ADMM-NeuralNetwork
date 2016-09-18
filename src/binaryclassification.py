import numpy as np
import time

from src.functions import mbhe
from src.neuralnetwork import NeuralNetwork
from src.commons import get_max_index, convert_binary_to_number

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


def epoch(net, trn_instance, tst_instance, train_iters=1, warm_iters=0, verbose=2):
    # An Epoch consists of two phases: training and testing (validation)
    st = time.time()
    for i in range(warm_iters):
        # Training without Lagrange multiplier update
        net.warmstart(trn_instance.samples, trn_instance.targets)
    for i in range(train_iters):
        # Standard Training
        net.train(trn_instance.samples, trn_instance.targets)
    endt = time.time() - st

    flag = trn_instance != tst_instance
    if flag:
        # Accuracy over training data
        trnres = net.feedforward(trn_instance.samples)
        n = trnres.shape[1]
        c = 0
        for i in range(n):
            output = get_max_index(trnres[:, i])
            label = convert_binary_to_number(trn_instance.targets[:, i],
                                             trn_instance.targets.shape[0])
            if output == label:
                c += 1
        trn_approx = float(c) / float(n)

    # Accuracy over validation data
    res = net.feedforward(tst_instance.samples)
    test = res.shape[1]
    c = 0
    for i in range(test):
        output = get_max_index(res[:, i])
        label = convert_binary_to_number(tst_instance.targets[:, i],
                                         tst_instance.targets.shape[0])
        if output == label:
            c += 1
    tst_approx = float(c) / float(test)
    residual = _residual(net.z[-1], net.w[-1], net.a[-1], net.beta)
    decs = 5

    if verbose == 2:
        print("Training time: %s sec" % np.round(endt, decimals=decs))
        loss = mbhe(res, tst_instance.targets)
        print("Mean Loss: %s" % str(np.round(loss, decimals=decs)))
        print("Residual: %s" % str(np.round(residual, decimals=decs)))
        print("Testing Results: %s/%s" % (c, test))
        if flag:
            print("Training Accuracy: %s" % str(np.round(trn_approx, decimals=decs)))
    if verbose != 0:
        if flag:
            print("Testing Accuracy: %s" % str(np.round(tst_approx, decimals=decs)))
        else:
            print("Accuracy: %s" % str(np.round(tst_approx, decimals=decs)))
        print("=============")
    return net, tst_approx, residual


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
