from copy import deepcopy

import numpy as np
import time

from sklearn import datasets
from sklearn.cross_validation import train_test_split

from src.neuralnetwork import Instance, NeuralNetwork


__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


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


def get_data():
    rng = 4
    cl = 2
    X, y = datasets.make_classification(n_samples=30000, n_features=128,
                                        n_informative=2, n_redundant=64,
                                        n_repeated=16, random_state=rng)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.5,
                                                        random_state=rng)

    trg_train = np.zeros((cl, len(y_train)), dtype='uint8')
    for e in range(trg_train.shape[1]):
        v = y_train[e]
        trg_train[v, e] = 1

    trg_test = np.zeros((cl, len(y_test)), dtype='uint8')
    for e in range(trg_test.shape[1]):
        v = y_test[e]
        trg_test[v, e] = 1

    trn = Instance(X_train.T, trg_train)
    tst = Instance(X_test.T, trg_test)
    return trn, tst


def get_iris(rng=42, tst_size=0.3):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=tst_size,
                                                        random_state=rng)

    trg_train = np.zeros((3, len(y_train)), dtype='uint8')
    for e in range(trg_train.shape[1]):
        v = y_train[e]
        trg_train[v, e] = 1

    trg_test = np.zeros((3, len(y_test)), dtype='uint8')
    for e in range(trg_test.shape[1]):
        v = y_test[e]
        trg_test[v, e] = 1

    trn = Instance(X_train.T, trg_train)
    tst = Instance(X_test.T, trg_test)
    return trn, tst


def column_normalisation(x):
    pass


def rnd_train(net, trn_instance, train_iters=1, warm_iters=0):
    st = time.time()
    for i in range(warm_iters):
        # Training without Lagrange multiplier update
        net.warmstart(trn_instance.samples, trn_instance.targets)
    for i in range(train_iters):
        # Standard Training
        net.train(trn_instance.samples, trn_instance.targets)
    endt = np.round((time.time() - st), decimals=6)
    return net, endt

from src.commons import get_max_index, convert_binary_to_number
def rnd_test(net, tst_instance):
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
    approx = float(c) / float(test)
    return np.round(approx, decimals=6)


def rnd_measure(accuracy, ws, m=10, k=30):
    trn, tst = get_data()

    rundict = {
        'runc': [0 for p in range(k)],
        'runover': 0,
        'timec': 0.
    }

    def residual(z, w, a, beta):
        lr = beta * (z - (np.dot(w, a)))
        e = np.mean(
            [lr[k, w] for k in range(z.shape[0]) for w in range(z.shape[1])],
            dtype=np.float64)
        return e

    for it in range(m):
        net = NeuralNetwork(trn.samples.shape[1],
                            trn.samples.shape[0],
                            trn.targets.shape[0],
                            96, gamma=10., beta=1.)

        flag = False
        ttmp = 0.
        g = np.random.randint(1000)
        mean = 0.
        print("============ " + str(it + 1))

        for innit in range(k):
            if flag is False:
                net, t = rnd_train(net, trn, train_iters=0, warm_iters=ws)
                acc = rnd_test(net, tst)
                # resid = residual(net.z[-1], net.w[-1], net.a[-1], net.beta)
                # print("Residual: %s" % str(resid))
                flag = True
                ttmp += t
            # mean = mean_lambda(net.l)
            if acc < accuracy:
                net, t = rnd_train(net, trn, train_iters=1, warm_iters=0)
                acc = rnd_test(net, tst)
                # resid = residual(net.z[-1], net.w[-1], net.a[-1], net.beta)
                ttmp += t
            else:
                rundict['runc'][innit] += 1
                rundict['timec'] += ttmp
                break
            """
            if mean_lambda(net.l) - mean < 0.000000001:
                print(mean_lambda(net.l))
                print(mean)
                print(acc)
                break
            """
            if innit == k - 1:
                rundict['runover'] += 1
                print("Reached Accuracy: %s" % str(acc))
            elif innit % 5 == 0:
                print("Accuracy at %s: %s" % (str(innit), str(acc)))
                #for h in range(30):
                #    print(net.l[0, h+g], end=' ')
                print()
                e = mean_lambda(net.l)
                #cn = comp_lambda_target(net.l, trn.targets)
                print("Lambda mean: %s" % str(e))
                #print("Lambda-targets divergence: %s" % str(cn))
                print("------------------")

    overs = rundict['runover'] / m
    runs = [x / m for x in rundict['runc']]
    if m > rundict['runover']:
        times = rundict['timec'] / (m - rundict['runover'])
    else:
        times = 0
    print(runs)
    print(overs)
    return runs, overs, times


def mean_lambda(l):
    e = np.mean(
        [np.abs(l[k, w]) for k in range(l.shape[0]) for w in range(l.shape[1])],
        dtype=np.float64)
    return e


def comp_lambda_target(l, y):
    counter = 0
    for i in range(l.shape[0]):
        for j in range(l.shape[1]):
            if y[i, j] == 0 and l[i, j] >= 0.:
                counter += 1
            elif y[i, j] == 1 and l[i, j] <= 0.:
                counter += 1
    return counter


def iris_measure(trn, tst, ws, m=10, k=100):
    res = []

    class rundict():
        def __init__(self, accuracylabel, timelabel, runsnumber):
            self.accuracy = accuracylabel
            self.time = timelabel
            self.run = runsnumber

    for it in range(m):
        net = NeuralNetwork(trn.samples.shape[1],
                            trn.samples.shape[0],
                            trn.targets.shape[0],
                            23, gamma=1., beta=0.5)

        flag = False
        ttmp = 0.
        tmpacc = -1.

        for innit in range(k):
            if flag is False:
                net, t = rnd_train(net, trn, train_iters=0, warm_iters=ws)
                acc = rnd_test(net, tst)
                flag = True
                ttmp += t

            if acc < 0.99:
                net, t = rnd_train(net, trn, train_iters=1, warm_iters=0)
                acc = rnd_test(net, tst)
                if tmpacc >= acc:
                    res.append(rundict(tmpacc, ttmp, innit))
                    break
                tmpacc = acc
            else:
                res.append(rundict(acc, ttmp, innit + 1))
                break
            if innit == k - 1:
                res.append(rundict(acc, ttmp, innit + 1))
    return res


def main_classification():
    import operator
    p = []
    a = [0.85]
    for i in a:
        runs, over, times = iris_measure(i, 1, k=100)
        if times == 0:
            continue
        else:
            times += ((i * 10) * over)
        # print(times)
        p.append({'x': times, 'y': i})
    newlist = sorted(p, key=operator.itemgetter('x'))
    # plotout(newlist)


def main_iris():

    def comp_iris(ming, maxg, mg, sp=0.3):
        mean_acc = 0.
        mean_time = 0.
        mean_runs = 0.
        min_acc = 0.
        max_acc = 0.
        delta = maxg - ming
        for i in range(ming, maxg):
            trn, tst = get_iris(rng=i, tst_size=sp)
            res = iris_measure(trn, tst, 1, m=mg)
            mean_acc += np.mean([e.accuracy for e in res])
            mean_time += np.mean([e.time for e in res])
            mean_runs += np.mean([e.run for e in res])
            min_acc += np.min([e.accuracy for e in res])
            max_acc += np.max([e.accuracy for e in res])
        print(
            "mean accuracy: %f   min peak: %f   max peak: %f   mean time: %f   mean runs: %f" %
            (mean_acc / delta, min_acc / delta, max_acc / delta, mean_time / delta, mean_runs / delta))

    print("===================================================================" * 2)
    print("Compare one execution with two different splitting of the dataset")
    trn, tst = get_iris(rng=11)
    res2 = iris_measure(trn, tst, 1, m=1)
    trn, tst = get_iris(rng=42)
    res1 = iris_measure(trn, tst, 1, m=1)
    print("rng: %d   accuracy: %f   time: %f   runs: %d" % (42, res1[0].accuracy, res1[0].time, res1[0].run))
    print("rng: %d   accuracy: %f   time: %f   runs: %d" % (11, res2[0].accuracy, res2[0].time, res2[0].run))
    print("===================================================================" * 2)
    print("Compare multiple executions of the same splitting (rng = 42)")
    comp_iris(42, 43, 1000)
    print("===================================================================" * 2)
    print("Compare multiple executions of different splitting of the dataset   [0 <= rng < 10]")
    comp_iris(0, 10, 100)
    print("===================================================================" * 2)
    print("Compare multiple executions of different splitting of the dataset", end="")
    print("   [random <= rng < random + 100]")

    g = 1 + np.random.randint(1000) + (np.random.randint(20) * np.random.randint(51))
    print("random = %s" % str(g))
    comp_iris(g, g + 100, 100)
    print("===================================================================" * 2)
    print("Compare multiple executions of different splitting of the dataset", end="")
    print("   [random <= rng < random + 100]", end="")
    print("   [0.1 <= test size <= 0.5]")
    g = 1 + np.random.randint(1000) + (np.random.randint(20) * np.random.randint(51))
    print("random = %s" % str(g))
    print("test size: %s   " % str(0.1), end="")
    comp_iris(g, g + 100, 100, sp=0.1)
    print("test size: %s   " % str(0.2), end="")
    comp_iris(g, g + 100, 100, sp=0.2)
    print("test size: %s   " % str(0.3), end="")
    comp_iris(g, g + 100, 100, sp=0.3)
    print("test size: %s   " % str(0.4), end="")
    comp_iris(g, g + 100, 100, sp=0.4)
    print("test size: %s   " % str(0.5), end="")
    comp_iris(g, g + 100, 100, sp=0.5)
    print("===================================================================" * 2)


if __name__ == '__main__':
    main_iris()
