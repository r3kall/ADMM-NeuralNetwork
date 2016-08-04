import pytest
import numpy
import numpy.matlib

from model.neuralnetwork import weight_update

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


@pytest.fixture(scope='module')
def weight():
    l0 = numpy.matlib.randn(1000, 1)
    l1 = numpy.matlib.randn(500, 1)
    l2 = numpy.matlib.randn(200, 1)
    l3 = numpy.matlib.randn(100, 1)
    l4 = numpy.matlib.randn(50, 1)
    l5 = numpy.matlib.randn(10, 1)
    return l0, l1, l2, l3, l4, l5


def test_weight_update_1(weight):
    assert weight_update(weight[0], weight[1]) is not None
    assert weight_update(weight[1], weight[2]) is not None
    assert weight_update(weight[2], weight[3]) is not None
    assert weight_update(weight[3], weight[4]) is not None
    assert weight_update(weight[4], weight[5]) is not None


def test_weight_update_2(weight):
    for i in range(10):
        assert weight_update(weight[0], weight[1]) is not None
        assert weight_update(weight[1], weight[2]) is not None
        assert weight_update(weight[2], weight[3]) is not None
        assert weight_update(weight[3], weight[4]) is not None
        assert weight_update(weight[4], weight[5]) is not None


def test_weight_update_3(weight):
    for i in range(100):
        assert weight_update(weight[0], weight[1]) is not None
        assert weight_update(weight[1], weight[2]) is not None
        assert weight_update(weight[2], weight[3]) is not None
        assert weight_update(weight[3], weight[4]) is not None
        assert weight_update(weight[4], weight[5]) is not None


def test_weight_update_4(weight):
    for i in range(1000):
        assert weight_update(weight[0], weight[1]) is not None
        assert weight_update(weight[1], weight[2]) is not None
        assert weight_update(weight[2], weight[3]) is not None
        assert weight_update(weight[3], weight[4]) is not None
        assert weight_update(weight[4], weight[5]) is not None


def test_weight_update_5(weight):
    for i in range(10000):
        assert weight_update(weight[0], weight[1]) is not None
        assert weight_update(weight[1], weight[2]) is not None
        assert weight_update(weight[2], weight[3]) is not None
        assert weight_update(weight[3], weight[4]) is not None
        assert weight_update(weight[4], weight[5]) is not None


def test_weight_update_6(weight):
    for i in range(100000):
        assert weight_update(weight[0], weight[1]) is not None
        assert weight_update(weight[1], weight[2]) is not None
        assert weight_update(weight[2], weight[3]) is not None
        assert weight_update(weight[3], weight[4]) is not None
        assert weight_update(weight[4], weight[5]) is not None
