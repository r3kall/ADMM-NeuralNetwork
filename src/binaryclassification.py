from copy import deepcopy

import numpy as np

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

    decs = 4
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
    return net, approx


def _residual(z, w, a, beta):
    lr = beta * (z - (np.dot(w, a)))
    e = np.mean([np.abs(lr[k, w]) for k in range(z.shape[0]) for w in range(z.shape[1])],
                dtype=np.float64)
    return e


def find_params(net, trn_instance, tst_instance, eps=1., tries=10, iters=4):
    count = tries
    gamma = 1.
    beta = 1.
    net0 = deepcopy(net)
    net1 = deepcopy(net)
    net0.gamma = gamma
    net0.beta = beta
    while count > 0:
        nn0, approx0 = epoch(deepcopy(net0),
                             trn_instance, tst_instance,
                             train_iters=iters, warm_iters=iters, verbose=0)
        net0.gamma += eps
        nn, approx = epoch(deepcopy(net0),
                           trn_instance, tst_instance,
                           train_iters=iters, warm_iters=iters, verbose=0)
        if approx0 < approx:
            gamma = net0.gamma
            count -= 1
        else:
            break

    net1.gamma = gamma
    net1.beta = beta
    count = 10
    while count > 0:
        nn0, approx0 = epoch(deepcopy(net1),
                             trn_instance, tst_instance,
                             train_iters=iters, warm_iters=iters, verbose=0)
        net1.beta += eps
        nn, approx = epoch(deepcopy(net1),
                           trn_instance, tst_instance,
                           train_iters=iters, warm_iters=iters, verbose=0)
        if approx0 < approx:
            beta = net1.beta
            count -= 1
        else:
            break
    return gamma, beta


def binary_classification(net, trn_instance, tst_instance, *layers,
                          accuracy=0.95, warm_iters=4, verbose=2,
                          findk=False, adaptive=False):

    if net is None:
        net = NeuralNetwork(trn_instance.samples.shape[1], trn_instance.samples.shape[0],
                            trn_instance.targets.shape[0], *layers)

    if findk:
        gamma, beta = find_params(net, trn_instance, tst_instance)
        net.gamma = gamma
        net.beta = beta

    net, approx = epoch(net, trn_instance, tst_instance,
                        train_iters=0, warm_iters=warm_iters, verbose=verbose)

    while approx < accuracy:
        net, approx = epoch(net, trn_instance, tst_instance, verbose=verbose)

    return net
