import numpy as np
import time

from sklearn import datasets
from sklearn.cross_validation import train_test_split

from src.neuralnetwork import Instance, NeuralNetwork
from src.neuraltools import get_sub_instance
from src.binaryclassification import epoch


__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


def digits_load(classes, perc=75, sf=True):
    dataset = datasets.load_digits(n_class=classes)
    dim = dataset.target.shape[1]
    targets = np.zeros((classes, dim), dtype='uint8')

    for w in range(dim):
        v = dataset.target[w]
        targets[v, w] = 1

    trn = Instance(dataset.data.T, targets)
    tst = get_sub_instance(trn, percentage=perc, shuffle=sf)
    return trn, tst


def digits_measure(classes, accuracy, ws, m=10, k=20):
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

    for it in range(m):
        net = NeuralNetwork(vspace, indim, outdim, 75, gamma=10., beta=1.)
        flag = False
        ttmp = 0.
        for w in range(k):
            if flag is False:
                st = time.time()
                net, acc, resid = epoch(net, trn, trn, train_iters=1, warm_iters=ws, verbose=0)
                endt = time.time() - st
                flag = True
            else:
                st = time.time()
                net, acc, resid = epoch(net, trn, trn, train_iters=1, warm_iters=0, verbose=0)
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


def iris_load():
    iris = datasets.load_iris()
    targets = np.zeros((3, 150), dtype='float64')
    for e in range(150):
        v = iris.target[e]
        targets[v, e] = 1
    ist = Instance(iris.data.T, targets)
    return ist


def iris_measure(accuracy, ws, m=10, k=30):
    assert 0. < accuracy <= 1.
    ist = iris_load()

    rundict = {
        'runc' : [0 for p in range(k)],
        'runover' : 0,
        'timec' : 0.
    }

    for it in range(m):
        net = NeuralNetwork(150, 4, 3, 72, gamma=10., beta=1.)
        flag = False
        ttmp = 0.
        for w in range(k):
            if flag is False:
                st = time.time()
                net, acc, resid = epoch(net, ist, ist, train_iters=1, warm_iters=ws, verbose=0)
                endt = time.time() - st
                flag = True
            else:
                st = time.time()
                net, acc, resid = epoch(net, ist, ist, train_iters=1, warm_iters=0, verbose=0)
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


def maind():
    import operator
    p = []
    a = [0.945, 0.95]
    for i in a:
        runs, over, times = digits_measure(10, i, 10)
        if times == 0:
            continue
        else:
            times += ((i * 10) * over)
        print(times)
        p.append({'x': times, 'y': i})
    newlist = sorted(p, key=operator.itemgetter('x'))
    #plotout(newlist)


def get_data():
    X, y = datasets.make_classification(n_samples=10000, n_features=32,
                                        n_informative=2, n_redundant=16,
                                        n_repeated=8, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                        random_state=42)

    trg_train = np.zeros((2, len(y_train)), dtype='uint8')
    for e in range(trg_train.shape[1]):
        v = y_train[e]
        trg_train[v, e] = 1

    # print(y_train[0])
    # print(trg_train[:, 0])

    trg_test = np.zeros((2, len(y_test)), dtype='uint8')
    for e in range(trg_test.shape[1]):
        v = y_test[e]
        trg_test[v, e] = 1

    trn = Instance(X_train.T, trg_train)
    tst = Instance(X_test.T, trg_test)
    return trn, tst


def classification_measure(accuracy, ws, m=10, k=50):
    trn, tst = get_data()

    rundict = {
        'runc': [0 for p in range(k)],
        'runover': 0,
        'timec': 0.
    }

    for it in range(m):
        net = NeuralNetwork(trn.samples.shape[1],
                            trn.samples.shape[0],
                            trn.targets.shape[0],
                            128, gamma=10., beta=1.)
        flag = False
        ttmp = 0.
        print("Start training")
        for w in range(k):
            if flag is False:
                st = time.time()
                net, acc, resid = epoch(net, trn, tst, train_iters=1, warm_iters=ws, verbose=1)
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
    runs = [x/m for x in rundict['runc']]
    if m > rundict['runover']:
        times = rundict['timec'] / (m - rundict['runover'])
    else:
        times = 0
    print(runs)
    print(overs)
    return runs, overs, times


def main_classification():
    import operator
    p = []
    a = [0.93, 0.94]
    for i in a:
        runs, over, times = classification_measure(i, 20, k=30)
        if times == 0:
            continue
        else:
            times += ((i * 10) * over)
        print(times)
        p.append({'x': times, 'y': i})
    newlist = sorted(p, key=operator.itemgetter('x'))
    # plotout(newlist)


if __name__ == '__main__':
    main_classification()
