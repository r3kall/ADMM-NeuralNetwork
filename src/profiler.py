import numpy as np
import time

from sklearn import datasets
from sklearn.model_selection import train_test_split

from .neuralnetwork import Instance, NeuralNetwork
from .commons import get_max_index, convert_binary_to_number


__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


def train(net, trn_instance, train_iters=1, warm_iters=0):
    st = time.time()
    for i in range(warm_iters):
        # Training without Lagrange multiplier update
        net.warmstart(trn_instance.samples, trn_instance.targets)
    for i in range(train_iters):
        # Standard Training
        net.train(trn_instance.samples, trn_instance.targets)
    endt = np.round((time.time() - st), decimals=6)
    return net, endt


def test(net, tst_instance):
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


def accuracy_listing(runlist):
    r = []
    for e in runlist:
        r.append(np.round(e.accuracy, decimals=2))
    return r


def draw_histogram(l, dataname):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from scipy import stats

    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(9)
    fr = fig.patch
    fr.set_facecolor('white')

    data = np.array([int(i * 100) for i in l])
    lnspc = np.linspace(min(data), max(data), len(data))

    d = np.diff(np.unique(data)).min()
    left_of_first_bin = data.min() - float(d) / 2
    right_of_last_bin = data.max() + float(d) / 2
    plt.hist(data, np.arange(left_of_first_bin, right_of_last_bin + d, (d / 2)),
             normed=True, align='left', facecolor='g', alpha=0.6)

    w = 2.
    k = float(d) * 5.

    m, s = stats.norm.fit(data)
    print("mean: %f   sigma: %f" % (m, s))
    pdf_g = stats.norm.pdf(lnspc, m, s) * k
    plt.plot(lnspc, pdf_g, 'r--', label='Norm', linewidth=w)

    ag, bg, cg = stats.gamma.fit(data)
    pdf_gamma = stats.gamma.pdf(lnspc, ag, bg, cg) * k
    plt.plot(lnspc, pdf_gamma, 'b--', label="Gamma", linewidth=w)

    ab, bb, cb, db = stats.beta.fit(data)
    pdf_beta = stats.beta.pdf(lnspc, ab, bb, cb, db) * k
    plt.plot(lnspc, pdf_beta, 'k--', label="Beta", linewidth=w)

    normal = mpatches.Patch(color='red', label='Normal')
    gamma = mpatches.Patch(color='blue', label='Gamma')
    beta = mpatches.Patch(color='black', label='Beta')
    plt.legend(loc=2, handles=[normal, gamma, beta])

    plt.xlim(70, 102)
    plt.subplots_adjust(left=0.15)
    plt.grid(True)
    plt.xlabel("accuracy")
    plt.ylabel("probability")
    plt.title(r'{0} dataset  $ \ \mu={1}, \ \sigma={2}$'.format(dataname,
                                                              np.round(m, decimals=1),
                                                              np.round(s, decimals=1)))
    plt.show()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 DIGITS Dataset                                         #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def get_digits(classes=10, rng=42):
    X, y = datasets.load_digits(n_class=classes, return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
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

        net, t = train(net, trn, train_iters=0, warm_iters=ws)
        acc = test(net, tst)
        ttmp = t
        ctrl = 0
        tacc = -1.

        for innit in range(k):
            if acc < 0.99:
                net, t = train(net, trn, train_iters=1, warm_iters=0)
                acc = test(net, tst)
                ttmp += t

                if tacc >= acc:
                    ctrl += 1
                else:
                    tacc = acc
                    ttim = ttmp
                    tk = innit
                    ctrl = 0
                if ctrl >= 4:
                    res.append(rundict(tacc, ttim, tk + 1))
                    break
            else:
                res.append(rundict(acc, ttmp, innit + 1))
                break

            if innit == k - 1:
                if ctrl == 0:
                    res.append(rundict(acc, ttmp, innit + 1))
                else:
                    res.append(rundict(tacc, ttim, tk + 1))
    return res


def digits_fitting(m=10, k=100):
    accuracy = [[] for i in range(m)]
    validation = [[] for i in range(m)]
    timelabel = [[] for i in range(m)]

    for i in range(m):
        g = np.random.randint(2200)
        trn, tst = get_digits(rng=g)
        net = NeuralNetwork(trn.samples.shape[1],
                            trn.samples.shape[0],
                            trn.targets.shape[0],
                            129, gamma=10., beta=1.)

        net, t = train(net, trn, train_iters=0, warm_iters=10)
        val = test(net, trn)
        acc = test(net, tst)
        ttmp = t
        accuracy[i].append(acc)
        validation[i].append(val)
        timelabel[i].append(ttmp)

        for innit in range(k):
            net, t = train(net, trn, train_iters=1, warm_iters=0)
            val = test(net, trn)
            acc = test(net, tst)
            ttmp += t
            accuracy[i].append(acc)
            validation[i].append(val)
            timelabel[i].append(ttmp)

    al = []
    vl = []
    tl = []
    for j in range(k):
        a_item = np.mean([accuracy[i][j] for i in range(m)])
        v_item = np.mean([validation[i][j] for i in range(m)])
        t_item = np.mean([timelabel[i][j] for i in range(m)])
        al.append(a_item)
        vl.append(v_item)
        tl.append(t_item)
    al.insert(0, al[0] / 2)
    vl.insert(0, vl[0] / 2)
    tl.insert(0, tl[0] / 2)
    al.insert(0, al[0] / 4)
    vl.insert(0, vl[0] / 4)
    tl.insert(0, tl[0] / 4)
    al.insert(0, 0.)
    vl.insert(0, 0.)
    tl.insert(0, 0.)

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.interpolate import spline

    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(9)
    fr = fig.patch
    fr.set_facecolor('white')

    x = np.array(tl)
    x_sm = np.linspace(x.min(), x.max(), 10)
    y1 = spline(tl, al, x_sm)
    y2 = spline(tl, vl, x_sm)

    acc_line = mpatches.Patch(color='red', label='testing accuracy')
    val_line = mpatches.Patch(color='blue', label='validation accuracy')
    plt.legend(handles=[acc_line, val_line], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    plt.plot(x_sm, y1, 'r', x_sm, y2, 'b')

    plt.subplots_adjust(left=0.15)
    plt.xlabel("seconds")
    plt.ylabel("accuracy")
    plt.xlim(0, x.max())
    plt.ylim(0.80, 1.)
    plt.show()


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

    print("=" * 72)
    print("Compare one execution of one splitting (rng = 42)")
    comp_digits(42, 43, 10, 1)
    print("=" * 72)
    print("Compare multiple executions of the same splitting (rng = 42)")
    comp_digits(42, 43, 10, 100)
    print("=" * 72)
    print("Compare multiple executions of different splitting of the dataset", end="")
    print("   [random <= rng < random + 10]")
    g = 1 + np.random.randint(1000) + (np.random.randint(20) * np.random.randint(51))
    print("random = %s" % str(g))
    comp_digits(g, g + 10, 10, 10)
    print("=" * 72)
    print("Compare multiple executions of different splitting of the dataset", end="")
    print("   [random <= rng < random + 100]")
    g = 1 + np.random.randint(1000) + (np.random.randint(20) * np.random.randint(51))
    print("random = %s" % str(g))
    comp_digits(g, g + 100, 10, 10)
    print("=" * 72)


def digits_draw(interv, reps):
    res = []

    g = np.random.randint(1000) + np.random.randint(1000)
    f = g + interv
    print("seed: %d" % g)

    st = time.time()
    for i in range(g, f):
        trn, tst = get_digits(classes=10, rng=i)
        res += digits_measure(trn, tst, 12, m=reps)
    endt = time.time() - st
    print("Time: %s" % str(round(endt, ndigits=2)))

    l = accuracy_listing(res)
    draw_histogram(l, 'digits')


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
#                                 IRIS Dataset                                           #
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


def get_iris(rng=42, tst_size=0.3):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = iris_normalisation(X)

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


def iris_normalisation(x):
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
                net, t = train(net, trn, train_iters=0, warm_iters=ws)
                acc = test(net, tst)
                flag = True
                ttmp += t

            if acc < 0.99:
                net, t = train(net, trn, train_iters=1, warm_iters=0)
                acc = test(net, tst)
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


def iris_fitting(m=100, k=100):
    accuracy = [[] for i in range(m)]
    validation = [[] for i in range(m)]
    timelabel = [[] for i in range(m)]

    for i in range(m):
        g = np.random.randint(2200)
        trn, tst = get_iris(rng=g)
        net = NeuralNetwork(trn.samples.shape[1],
                            trn.samples.shape[0],
                            trn.targets.shape[0],
                            9, gamma=1., beta=0.5)

        net, t = train(net, trn, train_iters=0, warm_iters=1)
        acc = test(net, tst)
        ttmp = t

        for innit in range(k):
            net, t = train(net, trn, train_iters=1, warm_iters=0)
            val = test(net, trn)
            acc = test(net, tst)
            ttmp += t
            accuracy[i].append(acc)
            validation[i].append(val)
            timelabel[i].append(ttmp)

    al = []
    vl = []
    tl = []
    for j in range(k):
        a_item = np.mean([accuracy[i][j] for i in range(m)])
        v_item = np.mean([validation[i][j] for i in range(m)])
        t_item = np.mean([timelabel[i][j] for i in range(m)])
        al.append(a_item)
        vl.append(v_item)
        tl.append(t_item)


    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.interpolate import spline

    fig = plt.figure(1)
    fig.set_figheight(12)
    fig.set_figwidth(9)
    fr = fig.patch
    fr.set_facecolor('white')

    x = np.array(tl)
    x_sm = np.linspace(x.min(), x.max(), 10)
    y1 = spline(tl, al, x_sm)
    y2 = spline(tl, vl, x_sm)

    acc_line = mpatches.Patch(color='red', label='testing accuracy')
    val_line = mpatches.Patch(color='blue', label='validation accuracy')
    plt.legend(handles=[acc_line, val_line], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)

    plt.plot(x_sm, y1, 'r', x_sm, y2, 'b')

    plt.subplots_adjust(left=0.15)
    plt.xlabel("seconds")
    plt.ylabel("accuracy")
    plt.xlim(0, x.max())
    plt.ylim(0.80, 1.)
    plt.show()


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

    print("=" * 72)
    print("Compare multiple executions of the same splitting (rng = 42)")
    comp_iris(42, 43, 1000)
    print("=" * 72)
    print("Compare multiple executions of different splitting of the dataset", end="")
    print("   [random <= rng < random + 10]")
    g = 1 + np.random.randint(1000) + (np.random.randint(20) * np.random.randint(51))
    print("random = %s" % str(g))
    comp_iris(g, g + 10, 1000)
    print("=" * 72)
    print("Compare multiple executions of different splitting of the dataset", end="")
    print("   [random <= rng < random + 100]")
    g = 1 + np.random.randint(1000) + (np.random.randint(20) * np.random.randint(51))
    print("random = %s" % str(g))
    comp_iris(g, g + 100, 1000)
    print("=" * 72)


def iris_draw(interv, reps):
    res = []

    g = np.random.randint(1000) + np.random.randint(500)
    f = g + interv
    print("seed: %d" % g)

    st = time.time()
    for i in range(g, f):
        trn, tst = get_iris(rng=i)
        res += iris_measure(trn, tst, 1, m=reps)
    endt = time.time() - st
    print("Time: %s" % str(round(endt, ndigits=2)))

    l = accuracy_listing(res)
    draw_histogram(l, 'iris')
