from copy import deepcopy
import numpy as np
import time

from neuralnetwork import NeuralNetwork, Instance
from functions import cost, activation, evaluation
from logger import defineLogger, Loggers, Levels
import commons

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'

log = defineLogger(Loggers.STANDARD)
log.setLevel(Levels.INFO.value)


def epoch(nn, training_instance, test_instance, train_iter=1, warm_iter=4):
    st = time.time()
    for i in range(warm_iter):
        nn.warmstart(training_instance.samples, training_instance.targets)
    for i in range(train_iter):
        nn.train(training_instance.samples, training_instance.targets)
    endt = time.time() - st
    log.debug("Training time: %s" % np.round(endt, decimals=4))

    res = nn.feedforward(test_instance.samples)
    test = res.shape[1]
    loss = evaluation["mean_binary_loss"](res, test_instance.targets)
    log.debug("Mean Loss: %s" % str(np.round(loss, decimals=4)))
    c = 0
    for i in range(test):
        output = commons.get_max_index(res[:, i])
        label = commons.convert_binary_to_number(test_instance.targets[:, i], 10)
        if output == label:
            c += 1
    log.debug("Accuracy: %s/%s" % (c, test))
    approx = np.round(float(c) / float(test), decimals=4)
    log.debug("Approx: %s" % approx)
    log.debug("=============")
    return nn, approx
# end epoch


def adaptive_training(nn, samples, targets, tst_samples, tst_targets, threshold=0.95, interval=5):
    i = commons.omega(samples.shape[1], 10, 100)
    approx = 0.
    count = 0
    diff = 0.
    t = False
    while approx < threshold:
        tmp = approx
        if count % interval == 0 and count != 0:
            if diff / interval < 0.005 and not t:
                print("boost")
                i += 100
                t = True
            diff = 0
        elif threshold - approx <= 0.01:
            i = commons.minus(i, i // 100)
        elif threshold - approx <= 0.1:
            i = commons.minus(i, i // 10)
        elif threshold - approx <= 0.5:
            i = commons.minus(i, i // 3)
        nn, approx = epoch(nn, samples, targets,
                           tst_samples, tst_targets, warm_iter=i)
        diff += approx - tmp
        count += 1
        print(i)
    print(count)
    return nn, approx


def find_params(nn, samples, targets, tst_samples, tst_targets, eps):
    nn0, approx = epoch(deepcopy(nn), samples, targets, tst_samples, tst_targets)
    count = 10
    gamma = nn.gamma
    v = 0
    while count > 0 and gamma - eps > 0:
        if v >= 0:
            nn.gamma = gamma + eps
            nnp, p = epoch(deepcopy(nn), samples, targets, tst_samples, tst_targets)
        if v <= 0:
            nn.gamma = gamma - eps
            nnt, t = epoch(deepcopy(nn), samples, targets, tst_samples, tst_targets)
        if approx >= p and approx >= t:
            return gamma
        if p > t:
            gamma = gamma + eps
            approx = p
            v = 1
        else:
            gamma = gamma - eps
            approx = t
            v = -1
        count -= 1
    return gamma
