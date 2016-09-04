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


def epoch(nn, trn_instance, tst_instance, out_eval, label_eval,
          train_iter=1, warm_iter=1, error=evaluation["mean_binary_loss"]):

    st = time.time()
    for i in range(warm_iter):
        nn.warmstart(trn_instance.samples, trn_instance.targets)

    for i in range(train_iter):
        nn.train(trn_instance.samples, trn_instance.targets)
    endt = time.time() - st
    log.debug("Training time: %s" % np.round(endt, decimals=4))

    out  = nn.feedforward(tst_instance.samples)
    loss = error(out, tst_instance.targets)
    log.debug("Mean Loss: %s" % str(np.round(loss, decimals=4)))

    c    = 0
    test = out.shape[1]
    for i in range(test):
        output = out_eval(out[:, i])
        label  = label_eval(tst_instance.targets[:, i])
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
                i += commons.omega(samples.shape[1], 10, 100)
                t = True
            diff = 0
        else:
            i = commons.minus(i, i // 10)
        nn, approx = epoch(nn, samples, targets,
                           tst_samples, tst_targets, warm_iter=i)
        diff += approx - tmp
        count += 1
    return nn, approx


def find_params(nn, samples, targets, tst_samples, tst_targets, eps=1, iter=10):
    nn0, approx = epoch(deepcopy(nn), samples, targets, tst_samples, tst_targets)
    count = iter
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
            gamma += eps
            approx = p
            v = 1
        else:
            gamma -= eps
            approx = t
            v = -1
        count -= 1
    return gamma
