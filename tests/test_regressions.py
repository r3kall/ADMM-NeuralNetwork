from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize_scalar

from src.algorithms.admm import activation_update
from src.commons import get_percentage
from src.cyth.argminc import argminc
from src.cyth.binarymin import binarymin
from src.functions import mbhe
from src.neuralnetwork import Instance, NeuralNetwork
from src.neuraltools import (get_sub_instance, split_instance,
                             load_network_from_file, save_network_to_file)
from src import profiler


def dataset(n=12):
    samples = np.vstack((np.arange(n), np.arange(n) + 1.0))
    targets = np.eye(2, dtype=np.uint8)[np.arange(n) % 2].T
    return Instance(samples, targets)


@pytest.mark.parametrize("beta", [0.1, 0.5, 1., 10.])
def test_hinge_matches_independent_optimizer(beta):
    random = np.random.default_rng(17)
    m = random.normal(size=(2, 30))
    eps = random.normal(size=m.shape)
    targets = random.integers(0, 2, size=m.shape, dtype=np.uint8)
    z = binarymin(targets, eps, m, beta)
    for index in np.ndindex(m.shape):
        y, mu, dual = targets[index], m[index], eps[index]
        def objective(x):
            return max(0, 1 - x if y else x) + dual * x + beta * (x - mu)**2
        center = mu - dual / (2 * beta)
        result = minimize_scalar(objective, bounds=(min(center, 0)-2, max(center, 1)+2),
                                 method="bounded", options={"xatol": 1e-10})
        assert objective(z[index]) <= result.fun + 1e-8


def test_hinge_boundaries():
    y = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.uint8)
    m = np.array([[0., .2, .5, .5, .8, 1.]])
    assert_array_equal(binarymin(y, np.zeros_like(m), m, 1.), [[0, 0, 0, 1, 1, 1]])


@pytest.mark.parametrize("penalty", [0., -1., np.inf, np.nan])
def test_solver_rejects_invalid_penalties(penalty):
    x = np.zeros((1, 2))
    with pytest.raises(ValueError):
        binarymin(x.astype(np.uint8), x, x, penalty)
    with pytest.raises(ValueError):
        argminc(x, x, penalty, 1.)
    with pytest.raises(ValueError):
        argminc(x, x, 1., penalty)


def test_solver_rejects_bad_shapes_labels_and_nonfinite():
    x = np.zeros((1, 2))
    with pytest.raises(ValueError):
        binarymin(x.astype(np.uint8), x[:, :1], x, 1.)
    with pytest.raises(ValueError):
        binarymin(np.full((1, 2), 2, dtype=np.uint8), x, x, 1.)
    with pytest.raises(ValueError):
        binarymin(x.astype(np.uint8), x, np.full_like(x, np.nan), 1.)
    with pytest.raises(ValueError):
        argminc(x, x[:, :1], 1., 1.)


def test_relu_global_minimum():
    random = np.random.default_rng(5)
    a, m = random.normal(size=(2, 25, 25))
    gamma, beta = 10., .5
    z = argminc(a, m, gamma, beta)
    objective = lambda x: gamma * (a-np.maximum(0, x))**2 + beta * (x-m)**2
    negative = np.minimum(m, 0)
    positive = np.maximum((gamma*a+beta*m)/(gamma+beta), 0)
    assert_allclose(objective(z), np.minimum(objective(negative), objective(positive)))


def test_activation_satisfies_normal_equations():
    random = np.random.default_rng(13)
    w, z, h = random.normal(size=(3, 4)), random.normal(size=(3, 7)), random.normal(size=(4, 7))
    a = activation_update(w, z, h, .5, 2.)
    assert_allclose((.5 * w.T @ w + 2 * np.eye(4)) @ a, .5 * w.T @ z + 2 * h)


@pytest.mark.parametrize("layers", [(3,), (4, 3)])
def test_seeded_training_arrays_loss_and_checkpoint(tmp_path, layers):
    inst = dataset()
    net = NeuralNetwork(12, 2, 2, *layers, rng=42)
    other = NeuralNetwork(12, 2, 2, *layers, rng=42)
    for model in (net, other):
        model.warmstart(inst.samples, inst.targets)
        for _ in range(3):
            model.train(inst.samples, inst.targets)
    for a, b in zip(net.w + net.a + net.z, other.w + other.a + other.z):
        assert type(a) is np.ndarray
        assert np.isfinite(a).all()
        assert_array_equal(a, b)
    assert net.error(inst.samples, inst.targets) == mbhe(net.feedforward(inst.samples), inst.targets)
    path = tmp_path / "checkpoint.pkl"
    save_network_to_file(net, path)
    loaded = load_network_from_file(path)
    assert_array_equal(loaded.feedforward(inst.samples), net.feedforward(inst.samples))
    loaded.train(inst.samples, inst.targets)
    net.train(inst.samples, inst.targets)
    assert_array_equal(loaded.feedforward(inst.samples), net.feedforward(inst.samples))


def test_error_runs_inference_even_when_feature_and_class_shapes_match():
    inst = dataset()
    net = NeuralNetwork(12, 2, 2, 3, rng=1)
    assert net.error(inst.samples, inst.targets) == .5
    assert net.error(inst.samples, inst.targets) != mbhe(inst.samples, inst.targets)


@pytest.mark.parametrize("kwargs", [{"beta": 0}, {"gamma": -1}, {"beta": np.nan},
                                    {"gamma": np.inf}, {"code": "missing"}])
def test_invalid_constructor_settings(kwargs):
    with pytest.raises(ValueError):
        NeuralNetwork(12, 2, 2, 3, **kwargs)


@pytest.mark.parametrize("sizes", [(0, 2, 2, 3), (12, 0, 2, 3), (12, 2, 0, 3),
                                   (12, 2, 2, 0), (12, 2, 2, -1), (12, 2, 2, 1.5),
                                   (12, 2, 2, True), (12, 2, 2)])
def test_invalid_dimensions(sizes):
    with pytest.raises(ValueError):
        NeuralNetwork(*sizes)


@pytest.mark.parametrize("method", ["train", "warmstart"])
@pytest.mark.parametrize("invalid", ["sample_count", "features", "target_count", "classes", "fraction", "negative", "nonfinite", "multilabel"])
def test_bad_training_input_does_not_mutate_state(method, invalid):
    inst = dataset()
    net = NeuralNetwork(12, 2, 2, 3, rng=42)
    before = deepcopy(net)
    x, y = inst.samples.copy(), inst.targets.astype(float)
    if invalid == "sample_count": x = x[:, :-1]
    elif invalid == "features": x = x[:1]
    elif invalid == "target_count": y = y[:, :-1]
    elif invalid == "classes": y = np.vstack((y, np.zeros((1, 12))))
    elif invalid == "fraction": y[0, 0] = .5
    elif invalid == "negative": y[0, 0] = -1
    elif invalid == "nonfinite": x[0, 0] = np.nan
    elif invalid == "multilabel": y[:, 0] = 1
    with pytest.raises(ValueError):
        getattr(net, method)(x, y)
    for actual, old in zip(net.w + net.z + net.a + [net.l], before.w + before.z + before.a + [before.l]):
        assert_array_equal(actual, old)


def test_instance_checks_before_label_cast():
    inst = dataset()
    for y in (inst.targets[:, :-1], inst.targets.astype(np.int64) * 256, inst.targets * .5,
              np.zeros_like(inst.targets), np.ones_like(inst.targets)):
        with pytest.raises(ValueError):
            Instance(inst.samples, y)


@pytest.mark.parametrize("shuffle", [False, True])
def test_split_preserves_every_sample_pair_and_source(shuffle):
    inst = dataset()
    before = deepcopy(inst)
    remainder, subset = split_instance(inst, 25, shuffle, rng=42)
    assert remainder.samples.shape[1] == 9
    assert subset.samples.shape[1] == 3
    combined = np.concatenate((remainder.samples, subset.samples), axis=1)
    labels = np.concatenate((remainder.targets, subset.targets), axis=1)
    order = combined[0].argsort()
    assert_array_equal(combined[:, order], inst.samples)
    assert_array_equal(labels[:, order], inst.targets)
    assert_array_equal(inst.samples, before.samples)
    assert_array_equal(inst.targets, before.targets)
    subset.samples[:] = -1
    assert_array_equal(inst.samples, before.samples)
    sub1 = get_sub_instance(inst, 25, shuffle, rng=42)
    sub2 = get_sub_instance(inst, 25, shuffle, rng=42)
    assert_array_equal(sub1.samples, sub2.samples)


def test_percentages_and_empty_partitions():
    assert get_percentage(25, 7) == 1
    assert type(get_percentage(25, 7)) is int
    assert get_percentage(0, 7) == 0
    assert get_percentage(100, 7) == 7
    for pct in (-1, 101, np.nan):
        with pytest.raises(ValueError): get_percentage(pct, 7)
    with pytest.raises(ValueError): split_instance(dataset(), .1)
    with pytest.raises(ValueError): get_sub_instance(dataset(), 0)


def test_benchmark_test_labels_do_not_select_checkpoints(monkeypatch):
    trn, tst = profiler.get_iris()
    original_test = profiler.test
    heldout_calls = []
    def spy(net, instance):
        if instance is tst:
            heldout_calls.append(deepcopy(net))
        return original_test(net, instance)
    monkeypatch.setattr(profiler, "test", spy)
    first = profiler.iris_measure(trn, tst, 1, m=2, k=8, rng=42)
    assert len(heldout_calls) == 2
    selected = heldout_calls.copy()
    heldout_calls.clear()
    # Change every held-out label while retaining valid one-hot columns.
    tst.targets = np.roll(tst.targets, 1, axis=0)
    second = profiler.iris_measure(trn, tst, 1, m=2, k=8, rng=42)
    assert len(heldout_calls) == 2
    assert [(r.run, r.validation_accuracy) for r in first] == [(r.run, r.validation_accuracy) for r in second]
    for net1, net2 in zip(selected, heldout_calls):
        for w1, w2 in zip(net1.w, net2.w): assert_array_equal(w1, w2)


@pytest.mark.parametrize("dataset_name", ["iris", "digits"])
def test_fitting_plots_all_measured_points_only(monkeypatch, dataset_name):
    # Deterministic recorded values expose any fabricated or dropped points.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    calls = []
    monkeypatch.setattr(profiler, "train", lambda net, instance, **kw: (net, 2.))
    def score(net, instance):
        result = .5 + len(calls) * .01
        calls.append(result)
        return result
    monkeypatch.setattr(profiler, "test", score)
    fig = profiler._fitting(dataset_name, 1, 3, 42)
    lines = fig.axes[0].lines
    assert len(lines) == 2
    assert_array_equal(lines[0].get_xdata(), [2, 4, 6, 8])
    assert_array_equal(lines[0].get_ydata(), calls[::2])
    assert_array_equal(lines[1].get_ydata(), calls[1::2])
    assert lines[1].get_label() == "validation accuracy"
    plt.close(fig)


def test_histogram_accepts_constant_values():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = profiler.draw_histogram([.9, .9], "iris")
    assert fig.axes[0].patches
    plt.close(fig)


def test_helpers_import_without_loading_network_first():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c", "from src.neuraltools import split_instance"], check=True)


def test_zero_update_benchmark_and_invalid_counts():
    trn, tst = profiler.get_iris()
    results = profiler.iris_measure(trn, tst, 1, m=1, k=0)
    assert len(results) == 1
    assert results[0].run == 0
    for kwargs in ({"m": 0}, {"k": -1}, {"ws": -1}):
        with pytest.raises(ValueError):
            profiler.iris_measure(trn, tst, **({"ws": 1} | kwargs))
