import numpy as np
from sklearn import datasets
from argparse import ArgumentParser

from src.neuralnetwork import Instance
from src.binaryclassification import binary_classification

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


def digits(LAYERS, ACC, WS, AD, V):
    print()
    print("=============")
    dataset = datasets.load_digits()
    samples = dataset.data.T
    dim = samples.shape[1]
    targets = np.mat(np.zeros((10, dim), dtype='uint8'))

    for i in range(dim):
        v = dataset.target[i]
        targets[v, i] = 1

    ist = Instance(samples, targets)
    nn = binary_classification(ist, ist, LAYERS, accuracy=ACC,
                               warm_iters=WS, adaptive=AD, verbose=V)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-H", metavar="H", type=tuple, dest="layers", default=50)
    parser.add_argument("-th", metavar="THRESHOLD", type=float, dest="acc", default=0.95)
    parser.add_argument("-ws", metavar="WARMSTART", type=int, dest="ws", default=20)
    parser.add_argument("-ad", metavar="ADAPTIVE", type=bool, dest="ad", default=False)
    parser.add_argument("-v", metavar="VERBOSE", type=int, dest="v", default=2)

    args = parser.parse_args()
    digits(args.layers, args.acc, args.ws, args.ad, args.v)