
import numpy as np
import time

from data_processing import Mnist
from neuralnetwork import NeuralNetwork

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


def memprofile():
    trn = Mnist.getTrainingSet()
    #tst = Mnist.getTestingSet()
    nn = NeuralNetwork(60000, 784, 10, 200, beta=0.5, gamma=9.)
    nn.warmstart(trn['x'], trn['y'])
    #nn.train(trn['x'], trn['y'])


def main():
    memprofile()


if __name__ == "__main__":
    main()
