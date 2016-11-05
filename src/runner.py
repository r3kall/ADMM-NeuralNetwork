from argparse import ArgumentParser

from .profiler import main_iris, main_digits


__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


"""src.runner: provides entry point main()"""


def main():
    parser = ArgumentParser(description="Print data sets report")
    parser.add_argument('dataset', type=str, action='store', help='report of the data set')
    args = parser.parse_args()

    if args.dataset == 'iris':
        print('iris')
        main_iris()
    elif args.dataset == 'digits':
        print('digits')
        main_digits()
    else:
        print('Argument not valid, please choose between  [iris | digits]')
