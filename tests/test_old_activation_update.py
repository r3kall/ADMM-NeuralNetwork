import pytest
import numpy
import numpy.matlib

from model.layers import HiddenLayer

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


@pytest.fixture(scope='module')
def activation():
    l0 = HiddenLayer(1000, 500)
    l1 = HiddenLayer(500, 200)
    l2 = HiddenLayer(200, 100)
    l3 = HiddenLayer(100, 50)
    l4 = HiddenLayer(50, 10)
    return (l0, l1, l2, l3, l4)


def test_old_activation_update_1(activation):
    activation[0].calc_activation_array(1, activation[1].w, activation[1].z)
    activation[1].calc_activation_array(1, activation[2].w, activation[2].z)
    activation[2].calc_activation_array(1, activation[3].w, activation[3].z)
    activation[3].calc_activation_array(1, activation[4].w, activation[4].z)


def test_old_activation_update_2(activation):
    for i in range(10):
        activation[0].calc_activation_array(1, activation[1].w, activation[1].z)
        activation[1].calc_activation_array(1, activation[2].w, activation[2].z)
        activation[2].calc_activation_array(1, activation[3].w, activation[3].z)
        activation[3].calc_activation_array(1, activation[4].w, activation[4].z)


def test_old_activation_update_3(activation):
    for i in range(100):
        activation[0].calc_activation_array(1, activation[1].w, activation[1].z)
        activation[1].calc_activation_array(1, activation[2].w, activation[2].z)
        activation[2].calc_activation_array(1, activation[3].w, activation[3].z)
        activation[3].calc_activation_array(1, activation[4].w, activation[4].z)


def test_old_activation_update_4(activation):
    for i in range(1000):
        activation[0].calc_activation_array(1, activation[1].w, activation[1].z)
        activation[1].calc_activation_array(1, activation[2].w, activation[2].z)
        activation[2].calc_activation_array(1, activation[3].w, activation[3].z)
        activation[3].calc_activation_array(1, activation[4].w, activation[4].z)


def test_old_activation_update_5(activation):
    for i in range(10000):
        activation[0].calc_activation_array(1, activation[1].w, activation[1].z)
        activation[1].calc_activation_array(1, activation[2].w, activation[2].z)
        activation[2].calc_activation_array(1, activation[3].w, activation[3].z)
        activation[3].calc_activation_array(1, activation[4].w, activation[4].z)


def test_old_activation_update_6(activation):
    for i in range(100000):
        activation[0].calc_activation_array(1, activation[1].w, activation[1].z)
        activation[1].calc_activation_array(1, activation[2].w, activation[2].z)
        activation[2].calc_activation_array(1, activation[3].w, activation[3].z)
        activation[3].calc_activation_array(1, activation[4].w, activation[4].z)