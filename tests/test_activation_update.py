import pytest
import numpy
import numpy.matlib

from model.neuralnetwork import NeuralNetwork, activation_update

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


@pytest.fixture(scope='module')
def activation():
    nn = NeuralNetwork(1000, 10, 300)
    return nn.w, nn.z, nn.a


def test_activation_update_1(activation):
    activation_update(activation[0][1], activation[1][1], activation[2][0], 1, 10)
    #activation_update(activation[0][2], activation[1][2], activation[2][1], 1, 10)
    #activation_update(activation[0][3], activation[1][3], activation[2][2], 1, 10)
    #activation_update(activation[0][4], activation[1][4], activation[2][3], 1, 10)


def test_activation_update_2(activation):
    for i in range(10):
        activation_update(activation[0][1], activation[1][1], activation[2][0])
        #activation_update(activation[0][2], activation[1][2], activation[2][1], 1, 10)
        #activation_update(activation[0][3], activation[1][3], activation[2][2], 1, 10)
        #activation_update(activation[0][4], activation[1][4], activation[2][3], 1, 10)


def test_activation_update_3(activation):
    for i in range(100):
        activation_update(activation[0][1], activation[1][1], activation[2][0], 1, 10)
        #activation_update(activation[0][2], activation[1][2], activation[2][1], 1, 10)
        #activation_update(activation[0][3], activation[1][3], activation[2][2], 1, 10)
        #activation_update(activation[0][4], activation[1][4], activation[2][3], 1, 10)


def test_activation_update_4(activation):
    for i in range(1000):
        activation_update(activation[0][1], activation[1][1], activation[2][0], 1, 10)
        #activation_update(activation[0][2], activation[1][2], activation[2][1], 1, 10)
        #activation_update(activation[0][3], activation[1][3], activation[2][2], 1, 10)
        #activation_update(activation[0][4], activation[1][4], activation[2][3], 1, 10)


def test_activation_update_5(activation):
    for i in range(10000):
        activation_update(activation[0][1], activation[1][1], activation[2][0], 1, 10)
        #activation_update(activation[0][2], activation[1][2], activation[2][1], 1, 10)
        #activation_update(activation[0][3], activation[1][3], activation[2][2], 1, 10)
        #activation_update(activation[0][4], activation[1][4], activation[2][3], 1, 10)

"""
def test_activation_update_6(activation):
    for i in range(100000):
        activation_update(activation[0][1], activation[1][1], activation[2][0], 1, 10)
        activation_update(activation[0][2], activation[1][2], activation[2][1], 1, 10)
        activation_update(activation[0][3], activation[1][3], activation[2][2], 1, 10)
        activation_update(activation[0][4], activation[1][4], activation[2][3], 1, 10)
"""