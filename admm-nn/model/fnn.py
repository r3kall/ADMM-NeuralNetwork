import auxiliaries
from model.layers import InputLayer, HiddenLayer, LastLayer
import numpy as np

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


class FNN(object):
    def __init__(self, sample_length, number_of_classes, *hidden_layers):
        assert sample_length > 0
        assert number_of_classes > 0
        assert len(hidden_layers) > 0

        self.sample_length = sample_length
        self.number_of_classes = number_of_classes
        self.hidden_layers_list = []

        for index, dim in enumerate(hidden_layers):
            if index == 0:
                n_in = self.sample_length
            else:
                n_in = hidden_layers[index - 1]
            self.hidden_layers_list.append(HiddenLayer(n_in, dim))

        self.input_layer = InputLayer(self.sample_length, self.sample_length)
        self.last_layer = LastLayer(self.hidden_layers_list[-1].n_out,
                                    self.number_of_classes)

    def __str__(self):
        hl = ''
        for item in self.hidden_layers_list:
            hl = hl + item.__str__() + ' -> '
        return self.input_layer.__str__() + ' -> ' + hl + self.last_layer.__str__()

    def _train_single_hidden_layer(self):
        self.hidden_layers_list[0].train_layer(self.input_layer.a,
                                               self.last_layer.beta,
                                               self.last_layer.w,
                                               self.last_layer.z)

    def _train_hidden_layers(self):
        n = len(self.hidden_layers_list)
        for pos in range(n):
            if pos == 0:
                self.hidden_layers_list[pos].train_layer(self.input_layer.a,
                                                         self.hidden_layers_list[pos+1].beta,
                                                         self.hidden_layers_list[pos+1].w,
                                                         self.hidden_layers_list[pos+1].z)
            elif pos == n-1:
                self.hidden_layers_list[pos].train_layer(self.hidden_layers_list[pos-1].a,
                                                         self.last_layer.beta,
                                                         self.last_layer.w,
                                                         self.last_layer.z)
            else:
                self.hidden_layers_list[pos].train_layer(self.hidden_layers_list[pos-1].a,
                                                         self.hidden_layers_list[pos+1].beta,
                                                         self.hidden_layers_list[pos+1].w,
                                                         self.hidden_layers_list[pos+1].z)

    def _train(self, fun, samples, targets, n, n_of_layers):
        assert fun is not None
        for i in range(n):
            self.input_layer.layer_output(samples[i])
            for j in range(n_of_layers):
                fun()
            self.last_layer.train_layer(self.hidden_layers_list[-1].a, targets[i])

    def train(self, samples, targets, n):
        assert n > 0
        assert len(samples) == len(targets) == n
        n_of_layers = len(self.hidden_layers_list)
        if n_of_layers == 1:
            self._train(self._train_single_hidden_layer, samples, targets, n, 1)
        else:
            self._train(self._train_hidden_layers, samples, targets, n, n_of_layers)


def main():
    fnn = FNN(500, 10, 50, 30)
    c = 200
    samples, targets = auxiliaries.data_gen(500, 10, c)
    fnn.train(samples, targets, c)


if __name__ == "__main__":
    main()
