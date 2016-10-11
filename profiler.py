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
                            129, gamma=10., beta=2.)

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
                if ctrl >= 4:
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


def main_digits():
    def comp_digits(dmin, dmax, ws, dm, cl=10):
        mean_acc = 0.
        mean_time = 0.
        mean_runs = 0.
        min_acc = 0.
        max_acc = 0.
        delta = dmax - dmin
        for i in range(dmin, dmax):
            trn, tst = get_digits(classes=cl, rng=i)
            res = digits_measure(trn, tst, ws, m=dm)
            mean_acc += np.mean([e.accuracy for e in res])
            mean_time += np.mean([e.time for e in res])
            mean_runs += np.mean([e.run for e in res])
            min_acc += np.min([e.accuracy for e in res])
            max_acc += np.max([e.accuracy for e in res])
        print(
            "classes: %d  mean accuracy: %f  min peak: %f  "
            "max peak: %f  mean time: %f  mean runs: %f  warm iters: %d" %
            (cl, mean_acc / delta, min_acc / delta, max_acc / delta,
             mean_time / delta, mean_runs / delta, ws))

    print("===================================================================" * 2)
    print("Compare one execution with two different splitting of the dataset")
    trn, tst = get_digits(rng=11)
    res2 = digits_measure(trn, tst, 10, m=1)
    trn, tst = get_digits(rng=42)
    res1 = digits_measure(trn, tst, 10, m=1)
    print("rng: %d   accuracy: %f   time: %f   runs: %d" % (42, res1[0].accuracy, res1[0].time, res1[0].run))
    print("rng: %d   accuracy: %f   time: %f   runs: %d" % (11, res2[0].accuracy, res2[0].time, res2[0].run))
    print("===================================================================" * 2)
    print("Compare multiple executions of the same splitting (rng = 42)")
    comp_digits(42, 43, 10, 25)
    print("===================================================================" * 2)
    print("Compare multiple executions of different splitting of the dataset", end="")
    print("   [0 <= rng < 10]")
    comp_digits(0, 10, 10, 10)
    print("===================================================================" * 2)
    print("Compare multiple executions of different splitting of the dataset", end="")
    print("   [random <= rng < random + 100]")
    g = 1 + np.random.randint(1000) + (np.random.randint(20) * np.random.randint(51))
    print("random = %s" % str(g))
    comp_digits(g, g + 100, 10, 10)
    print("===================================================================" * 2)


def digits_accuracy_listing(runlist):
    r = []
    for e in runlist:
        r.append(np.round(e.accuracy, decimals=2))
    return r


def digits_histogram(l):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from scipy import stats

    data = np.array([int(i * 100) for i in l])
    nbins = 30
    print(len(data))

    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(9)
    fr = fig.patch
    fr.set_facecolor('white')

    plt.hist(data, nbins, normed=True, facecolor='g', alpha=0.75, align='right')
    lnspc = np.linspace(min(data), max(data), len(data))
    k = 2.

    w = 2.
    m, s = stats.norm.fit(data)
    pdf_g = stats.norm.pdf(lnspc, m, s) * k
    plt.plot(lnspc, pdf_g, 'r--', label='Norm', linewidth=w)

    # exactly same as above
    ag, bg, cg = stats.gamma.fit(data)
    pdf_gamma = stats.gamma.pdf(lnspc, ag, bg, cg) * k
    plt.plot(lnspc, pdf_gamma, 'b--', label="Gamma", linewidth=w)

    # guess what :)
    ab, bb, cb, db = stats.beta.fit(data)
    pdf_beta = stats.beta.pdf(lnspc, ab, bb, cb, db) * k
    plt.plot(lnspc, pdf_beta, 'k--', label="Beta", linewidth=w)

    normal = mpatches.Patch(color='red', label='Normal')
    gamma = mpatches.Patch(color='blue', label='Gamma')
    beta = mpatches.Patch(color='black', label='Beta')
    plt.legend(loc=2, handles=[normal, gamma, beta])

    plt.xlim(60, 102)
    plt.subplots_adjust(left=0.1)
    plt.grid(True)
    plt.xlabel("accuracy")
    plt.ylabel("probability density")
    plt.show()


def digits_draw(interv, reps):
    res = []

    g = np.random.randint(1000) + np.random.randint(1000)
    f = g + interv
    print("seed: %d" % g)

    st = time.time()
    for i in range(g, f):
        trn, tst = get_digits(classes=10, rng=i)
        res += digits_measure(trn, tst, 10, m=reps)
    endt = time.time() - st
    print("Time: %s" % str(round(endt, ndigits=2)))

    l = digits_accuracy_listing(res)
    digits_histogram(l)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 RANDOM Dataset                                         #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def get_random_data(rng=42):
    cl = 2
    X, y = datasets.make_classification(n_samples=3000, n_features=16,
                                        n_informative=2, n_redundant=4,
                                        n_repeated=2, random_state=rng)
    X = rnd_normalisation(X)

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


def rnd_normalisation(x):
    # x has shape (samples, features)
    for i in range(x.shape[0]):
        dem = np.sum(np.exp(x[i, :]))
        for j in range(x.shape[1]):
            x[i, j] = np.exp(x[i, j]) / dem
    return x


def rnd_measure(trn, tst, ws, m=10, k=100):
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
                            33, 9, gamma=3., beta=2.)

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
                if ctrl >= 4:
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
    def comp_random(dmin, dmax, ws, dm):
        mean_acc = 0.
        mean_time = 0.
        mean_runs = 0.
        min_acc = 0.
        max_acc = 0.
        delta = dmax - dmin
        for i in range(dmin, dmax):
            trn, tst = get_random_data(rng=i)
            res = rnd_measure(trn, tst, ws, m=dm)
            mean_acc += np.mean([e.accuracy for e in res])
            mean_time += np.mean([e.time for e in res])
            mean_runs += np.mean([e.run for e in res])
            min_acc += np.min([e.accuracy for e in res])
            max_acc += np.max([e.accuracy for e in res])
        print(
            "mean accuracy: %f  min peak: %f  "
            "max peak: %f  mean time: %f  mean runs: %f  warm iters: %d" %
            (mean_acc / delta, min_acc / delta, max_acc / delta,
             mean_time / delta, mean_runs / delta, ws))

    its = 3
    print("=" * 72)
    print("Compare multiple executions of the same splitting (rng = 42)")
    comp_random(42, 43, its, 5)
    print("=" * 72)
    print("Compare multiple executions of different splitting of the dataset", end="")
    print("   [0 <= rng < 10]")
    comp_random(0, 10, its, 5)
    print("=" * 72)
    print("Compare multiple executions of different splitting of the dataset", end="")
    print("   [random <= rng < random + 100]")
    g = 1 + np.random.randint(1000) + (np.random.randint(20) * np.random.randint(51))
    print("random = %s" % str(g))

    print("=" * 72)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 IRIS Dataset                                           #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def get_iris(rng=42, tst_size=0.25):
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


def iris_accuracy_listing(runlist):
    r = []
    for e in runlist:
        r.append(np.round(e.accuracy, decimals=2))
    return r


def iris_histogram(l):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from scipy import stats

    data = np.array([int(i * 100) for i in l])
    nbins = 75
    print(len(data))

    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(9)
    fr = fig.patch
    fr.set_facecolor('white')

    plt.hist(data, nbins, normed=True, facecolor='g', alpha=0.75, align='right')
    lnspc = np.linspace(min(data), max(data), len(data))
    k = 3.5

    w = 2.
    m, s = stats.norm.fit(data)
    pdf_g = stats.norm.pdf(lnspc, m, s) * k
    plt.plot(lnspc, pdf_g, 'r--', label='Norm', linewidth=w)

    # exactly same as above
    ag, bg, cg = stats.gamma.fit(data)
    pdf_gamma = stats.gamma.pdf(lnspc, ag, bg, cg) * k
    plt.plot(lnspc, pdf_gamma, 'b--', label="Gamma", linewidth=w)

    # guess what :)
    ab, bb, cb, db = stats.beta.fit(data)
    pdf_beta = stats.beta.pdf(lnspc, ab, bb, cb, db) * k
    plt.plot(lnspc, pdf_beta, 'k--', label="Beta", linewidth=w)

    normal = mpatches.Patch(color='red', label='Normal')
    gamma = mpatches.Patch(color='blue', label='Gamma')
    beta = mpatches.Patch(color='black', label='Beta')
    plt.legend(loc=2, handles=[normal, gamma, beta])

    plt.xlim(60, 102)
    plt.subplots_adjust(left=0.1)
    plt.grid(True)
    plt.xlabel("accuracy")
    plt.ylabel("probability density")
    plt.show()


def iris_draw(interv, reps):
    res = []

    g = np.random.randint(1000) + np.random.randint(1000)
    f = g + interv
    print("seed: %d" % g)

    st = time.time()
    for i in range(g, f):
        trn, tst = get_iris(rng=i)
        res += iris_measure(trn, tst, 1, m=reps)
    endt = time.time() - st
    print("Time: %s" % str(round(endt, ndigits=2)))

    l = iris_accuracy_listing(res)
    iris_histogram(l)


if __name__ == '__main__':
    main_random()
