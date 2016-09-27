import numpy as np
import time

from sklearn import datasets

from src.neuralnetwork import Instance, NeuralNetwork
from src.neuraltools import get_sub_instance
from src.binaryclassification import epoch


__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


def digits_load(classes, perc=50, sf=True):
    n = classes
    dataset = datasets.load_digits(n_class=n)
    samples = dataset.data.T
    dim = samples.shape[1]
    targets = np.mat(np.zeros((n, dim), dtype='uint8'))

    for i in range(dim):
        v = dataset.target[i]
        targets[v, i] = 1

    trn = Instance(samples, targets)
    tst = get_sub_instance(trn, percentage=perc, shuffle=sf)
    return trn, tst


def digits_measure(classes, accuracy, m, k, ws):
    assert 1 < classes <= 10
    assert 0. < accuracy <= 1.
    trn, tst = digits_load(classes)

    rundict = {
        'runc' : [0 for p in range(k)],
        'runover' : 0,
        'timec' : 0.
    }

    indim = trn.samples.shape[0]
    outdim = trn.targets.shape[0]
    vspace = trn.samples.shape[1]

    for i in range(m):
        net = NeuralNetwork(vspace, indim, outdim, 128, gamma=10., beta=2.)
        flag = False
        ttmp = 0.
        for w in range(k):
            if flag is False:
                st = time.time()
                net, acc, resid = epoch(net, trn, tst, train_iters=1, warm_iters=ws, verbose=0)
                endt = time.time() - st
                flag = True
            else:
                st = time.time()
                net, acc, resid = epoch(net, trn, tst, train_iters=1, warm_iters=0, verbose=0)
                endt = time.time() - st
            ttmp += endt
            if acc >= accuracy:
                rundict['runc'][w] += 1
                rundict['timec'] += ttmp
                break
            if w == k - 1:
                rundict['runover'] += 1
                print("ACC: %s" % str(acc))

    overs = rundict['runover'] / m
    runs = [x / (m) for x in rundict['runc']]
    if m > rundict['runover']:
        times = rundict['timec'] / (m - rundict['runover'])
    else:
        times = 0
    print(runs)
    print(overs)
    return runs, overs, times


def plotout(pairs):
    import matplotlib.pyplot as plt
    x = [e['x'] for e in pairs]
    y = [e['y'] for e in pairs]
    line = plt.plot(x, y)
    plt.setp(line, color='r', linewidth=1.5)
    #plt.xticks(np.arange(min(x), max(x) + 1, 0.5))
    #plt.yticks(np.arange(0.65, 1., 0.05))
    plt.ylim(min(y), 1.)
    plt.xlim(min(x), max(x))
    plt.show()


if __name__ == '__main__':
    import operator
    p = []
    a = [0.945, 0.95]
    for i in a:
        runs, over, times = digits_measure(10, i, 100, 50, 20)
        if times == 0:
            continue
        else:
            times += ((i * 10) * over)
        print(times)
        p.append({'x': times, 'y': i})
    newlist = sorted(p, key=operator.itemgetter('x'))
    plotout(newlist)
