"""Reproducible benchmarks with validation-only checkpoint selection."""
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from .neuralnetwork import Instance, NeuralNetwork


def _count(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def train(net, trn_instance, train_iters=1, warm_iters=0):
    _count(train_iters, "train_iters")
    _count(warm_iters, "warm_iters")
    start = perf_counter()
    for _ in range(warm_iters):
        net.warmstart(trn_instance.samples, trn_instance.targets)
    for _ in range(train_iters):
        net.train(trn_instance.samples, trn_instance.targets)
    return net, perf_counter() - start


def test(net, tst_instance):
    output = net.feedforward(tst_instance.samples)
    if output.shape != tst_instance.targets.shape:
        raise ValueError("target shape must match network output")
    return float(np.mean(output.argmax(axis=0) == tst_instance.targets.argmax(axis=0)))


def _instance(x, labels, classes):
    return Instance(x.T, np.eye(classes, dtype=np.uint8)[labels].T)


def _dataset_split(x, y, classes, rng, tst_size):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=tst_size, random_state=rng, stratify=y)
    return _instance(x_train, y_train, classes), _instance(x_test, y_test, classes)


def get_digits(classes=10, rng=42):
    if isinstance(classes, (bool, np.bool_)) or not isinstance(classes, (int, np.integer)) or not 2 <= classes <= 10:
        raise ValueError("classes must be an integer between 2 and 10")
    x, y = datasets.load_digits(n_class=classes, return_X_y=True)
    return _dataset_split(x, y, classes, rng, 0.3)


def get_iris(rng=42, tst_size=0.3):
    iris = datasets.load_iris()
    return _dataset_split(iris_normalisation(iris.data), iris.target, 3, rng, tst_size)


def iris_normalisation(x):
    """Apply the original fixed transform without changing the input array."""
    x = np.asarray(x, dtype=np.float64)
    return 2.059999 / (1 + np.exp(-x)) - 1.070999


def _validation_split(instance, seed, validation_size=0.2):
    indices = np.arange(instance.samples.shape[1])
    fitting, validation = train_test_split(
        indices, test_size=validation_size, random_state=seed,
        stratify=instance.targets.argmax(axis=0))
    return tuple(Instance(instance.samples[:, ix], instance.targets[:, ix])
                 for ix in (fitting, validation))


def _network(instance, dataset, rng):
    width, beta, gamma = (9, 0.5, 1.) if dataset == "iris" else (129, 1., 10.)
    return NeuralNetwork(instance.samples.shape[1], instance.samples.shape[0],
                         instance.targets.shape[0], width, beta=beta, gamma=gamma, rng=rng)


@dataclass
class RunResult:
    accuracy: float
    time: float
    run: int
    validation_accuracy: float


def _measure(trn, tst, ws, m, k, dataset, rng):
    _count(m, "m", 1)
    _count(k, "k")
    _count(ws, "ws")
    random = np.random.default_rng(rng)
    seed = int(random.integers(2**31))
    fitting, validation = _validation_split(trn, seed)
    results = []
    for _ in range(m):
        net = _network(fitting, dataset, random)
        net, elapsed = train(net, fitting, train_iters=0, warm_iters=ws)
        best_accuracy = test(net, validation)
        best_net = deepcopy(net)
        best_run = 0
        stale = 0
        for iteration in range(1, k + 1):
            if best_accuracy >= 0.99:
                break
            net, duration = train(net, fitting)
            elapsed += duration
            score = test(net, validation)
            if score > best_accuracy:
                best_accuracy, best_run = score, iteration
                best_net = deepcopy(net)
                stale = 0
            else:
                stale += 1
            if stale >= 4:
                break
        # Test labels never affect training or checkpoint selection.
        results.append(RunResult(test(best_net, tst), elapsed, best_run, best_accuracy))
    return results


def digits_measure(trn, tst, ws, m=10, k=100, rng=42):
    return _measure(trn, tst, ws, m, k, "digits", rng)


def iris_measure(trn, tst, ws, m=10, k=100, rng=42):
    return _measure(trn, tst, ws, m, k, "iris", rng)


def _plot_curves(times, training, validation):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(times, training, label="training accuracy")
    ax.plot(times, validation, label="validation accuracy")
    ax.set(xlabel="training seconds", ylabel="accuracy", ylim=(0, 1))
    ax.legend()
    return fig


def _fitting(dataset, m, k, rng):
    """Plot measured warm-start plus k updates; never use held-out test labels."""
    _count(m, "m", 1)
    _count(k, "k")
    random = np.random.default_rng(rng)
    curves = []
    for _ in range(m):
        seed = int(random.integers(2**31))
        trn, _ = get_iris(rng=seed) if dataset == "iris" else get_digits(rng=seed)
        fitting, validation = _validation_split(trn, seed)
        net = _network(fitting, dataset, random)
        warm = 1 if dataset == "iris" else 10
        net, elapsed = train(net, fitting, train_iters=0, warm_iters=warm)
        points = [(elapsed, test(net, fitting), test(net, validation))]
        for _ in range(k):
            net, duration = train(net, fitting)
            elapsed += duration
            points.append((elapsed, test(net, fitting), test(net, validation)))
        curves.append(points)
    mean = np.mean(curves, axis=0)
    return _plot_curves(mean[:, 0], mean[:, 1], mean[:, 2])


def digits_fitting(m=10, k=100, rng=42):
    return _fitting("digits", m, k, rng)


def iris_fitting(m=100, k=100, rng=42):
    return _fitting("iris", m, k, rng)


def accuracy_listing(runlist):
    return [round(e.accuracy, 2) for e in runlist]


def draw_histogram(values, dataname):
    import matplotlib.pyplot as plt
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("histogram needs a nonempty finite sequence")
    fig, ax = plt.subplots()
    ax.hist(values, bins="auto", density=True)
    ax.set(xlabel="test accuracy", ylabel="density", title=f"{dataname} dataset")
    return fig


def _main(dataset, repetitions=10, iterations=100, seed=42):
    trn, tst = get_iris(rng=seed) if dataset == "iris" else get_digits(rng=seed)
    measure = iris_measure if dataset == "iris" else digits_measure
    results = measure(trn, tst, 1 if dataset == "iris" else 10,
                      m=repetitions, k=iterations, rng=seed)
    print(f"{dataset}: mean test accuracy={np.mean([r.accuracy for r in results]):.6f}, "
          f"mean training seconds={np.mean([r.time for r in results]):.6f}, "
          f"mean selected updates={np.mean([r.run for r in results]):.2f}, seed={seed}")
    return results


def main_iris(repetitions=10, iterations=100, seed=42):
    return _main("iris", repetitions, iterations, seed)


def main_digits(repetitions=10, iterations=100, seed=42):
    return _main("digits", repetitions, iterations, seed)


def _draw(dataset, interv, reps, rng):
    _count(interv, "interv", 1)
    _count(reps, "reps", 1)
    random = np.random.default_rng(rng)
    results = []
    for _ in range(interv):
        seed = int(random.integers(2**31))
        trn, tst = get_iris(rng=seed) if dataset == "iris" else get_digits(rng=seed)
        results.extend(_measure(trn, tst, 1 if dataset == "iris" else 12,
                                reps, 100, dataset, seed))
    return draw_histogram(accuracy_listing(results), dataset)


def digits_draw(interv, reps, rng=42):
    return _draw("digits", interv, reps, rng)


def iris_draw(interv, reps, rng=42):
    return _draw("iris", interv, reps, rng)
