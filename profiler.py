import numpy as np
import time

from sklearn import datasets
from sklearn.model_selection import train_test_split

from src.neuralnetwork import Instance, NeuralNetwork
from src.commons import get_max_index, convert_binary_to_number


__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


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


def get_digits(classes=10, rng=42):
    X, y = datasets.load_digits(n_class=classes, return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=rng)

    trg_train = np.zeros((classes, len(y_train)), dtype='uint8')
    for e in range(trg_train.shape[1]):
        v = y_train[e]
        trg_train[v, e] = 1

    trg_test = np.zeros((classes, len(y_test)), dtype='uint8')
    for e in range(trg_test.shape[1]):
        v = y_test[e]
        trg_test[v, e] = 1

    trn = Instance(X_train.T, trg_train)
    tst = Instance(X_test.T, trg_test)
    return trn, tst


def digits_measure(trn, tst, ws, m=10, k=100):
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
                            129, gamma=10., beta=1.)

        flag = False
        ttmp = 0.
        ctrl = 0
        tacc = -1.

        for innit in range(k):
            if flag is False:
                net, t = rnd_train(net, trn, train_iters=0, warm_iters=ws)
                acc = rnd_test(net, tst)
                flag = True
                ttmp += t

            if acc < 0.99:
                net, t = rnd_train(net, trn, train_iters=1, warm_iters=0)
                acc = rnd_test(net, tst)
                if tacc >= acc:
                    ctrl += 1
                else:
                    tacc = acc
                    ttim = ttmp + t
                    tk = innit
                    ctrl = 0
                if ctrl >= 7:
                    res.append(rundict(tacc, ttim, tk + 1))
                    break
                ttmp += t
            else:
                res.append(rundict(acc, ttmp, innit + 1))
                break

            if innit == k - 1:
                if ctrl == 0:
                    res.append(rundict(acc, ttmp, innit + 1))
                else:
                    res.append(rundict(tacc, ttim, tk + 1))
    return res


def digits_standard(runlist):
    from collections import defaultdict
    d = defaultdict(int)
    for e in runlist:
        a = np.round(e.accuracy, decimals=2)
        d[a] += 1
    return d


def digits_histogram(accuracy_dict):
    import matplotlib.pyplot as plt
    pass


def main_digits():
    trn, tst = get_digits()
    res = digits_measure(trn, tst, 10, m=5, k=100)
    mydict = digits_standard(res)
    digits_histogram(mydict)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 RANDOM Dataset                                         #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def get_random_data(rng=42):
    cl = 2
    X, y = datasets.make_classification(n_samples=10000, n_features=48,
                                        n_informative=2, n_redundant=20,
                                        n_repeated=2, random_state=rng)
    X = rnd_normalisation(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.6,
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


def rnd_normalisation(x):
    # x has shape (samples, features)
    for i in range(x.shape[0]):
        dem = np.sum(np.exp(x[i, :]))
        for j in range(x.shape[1]):
            x[i, j] = np.exp(x[i, j]) / dem
    return x


def rnd_measure(trn, tst, ws, m=10, k=50):
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
                            49, 25, gamma=10., beta=1.)

        flag = False
        ttmp = 0.
        ctrl = 0
        tacc = -1.

        for innit in range(k):
            if flag is False:
                net, t = rnd_train(net, trn, train_iters=0, warm_iters=ws)
                acc = rnd_test(net, tst)
                flag = True
                ttmp += t

            if acc < 0.99:
                net, t = rnd_train(net, trn, train_iters=1, warm_iters=0)
                acc = rnd_test(net, tst)
                if tacc >= acc:
                    ctrl += 1
                else:
                    tacc = acc
                    ttim = ttmp + t
                    tk = innit
                    ctrl = 0
                if ctrl >= 5:
                    res.append(rundict(tacc, ttim, tk + 1))
                    break
                ttmp += t
            else:
                res.append(rundict(acc, ttmp, innit + 1))
                break

            if innit == k - 1:
                if ctrl == 0:
                    res.append(rundict(acc, ttmp, innit + 1))
                else:
                    res.append(rundict(tacc, ttim, tk + 1))
    return res


def main_random():
    # import operator

    trn, tst = get_random_data(rng=42)
    res = rnd_measure(trn, tst, 10, m=5, k=50)
    for e in res:
        print(e.accuracy)

    # p.append({'x': times, 'y': i})
    # newlist = sorted(p, key=operator.itemgetter('x'))
    # plotout(newlist)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 IRIS Dataset                                           #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def get_iris(rng=42, tst_size=0.3):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = sigmoid_normalisation(X)

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


def sigmoid_normalisation(x):
    for i in range(150):
        for j in range(4):
            x[i, j] = (2.059999 * (1 / (1 + np.exp(- x[i, j])))) - 1.070999
    return x


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
                            9, gamma=1., beta=0.5)

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
                ttmp += t
                tmpacc = acc
            else:
                res.append(rundict(acc, ttmp, innit + 1))
                break
            if innit == k - 1:
                res.append(rundict(acc, ttmp, innit + 1))
    return res


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
    print("   [random <= rng < random + 10]", end="")
    print("   [0.1 <= test size <= 0.5]")
    g = 1 + np.random.randint(1000) + (np.random.randint(20) * np.random.randint(51))
    print("random = %s" % str(g))
    print("test size: %s   " % str(0.1), end="")
    comp_iris(g, g + 10, 100, sp=0.1)
    print("test size: %s   " % str(0.2), end="")
    comp_iris(g, g + 10, 100, sp=0.2)
    print("test size: %s   " % str(0.3), end="")
    comp_iris(g, g + 10, 100, sp=0.3)
    print("test size: %s   " % str(0.4), end="")
    comp_iris(g, g + 10, 100, sp=0.4)
    print("test size: %s   " % str(0.5), end="")
    comp_iris(g, g + 10, 100, sp=0.5)
    print("===================================================================" * 2)


if __name__ == '__main__':
    main_digits()
