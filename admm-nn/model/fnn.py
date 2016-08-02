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

    def _train(self, fun, samples, targets, n, n_of_layers, ws_iter):
        assert fun is not None
        for i in range(n):
            self.input_layer.layer_output(samples[i])
            for j in range(n_of_layers):
                fun()
            if ws_iter > 0:
                self.last_layer.warm_start(self.hidden_layers_list[-1].a, targets[i])
                ws_iter -= 1
            else:
                self.last_layer.train_layer(self.hidden_layers_list[-1].a, targets[i])

    def train(self, samples, targets, n, warmstart_percent=25):
        assert n > 0
        assert len(samples) == len(targets) == n
        ws = auxiliaries.get_percentage(warmstart_percent, n)
        n_of_layers = len(self.hidden_layers_list)
        if n_of_layers == 1:
            self._train(self._train_single_hidden_layer, samples, targets, n, 1, ws)
        else:
            self._train(self._train_hidden_layers, samples, targets, n, n_of_layers, ws)

    def _validate_single_hidden_layer(self):
        self.hidden_layers_list[0].layer_output(self.input_layer.a,
                                                self.last_layer.beta,
                                                self.last_layer.w,
                                                self.last_layer.z)

    def _validate_hidden_layers(self):
        n = len(self.hidden_layers_list)
        for pos in range(n):
            if pos == 0:
                self.hidden_layers_list[pos].layer_output(self.input_layer.a,
                                                          self.hidden_layers_list[pos+1].beta,
                                                          self.hidden_layers_list[pos+1].w,
                                                          self.hidden_layers_list[pos+1].z)
            elif pos == n-1:
                self.hidden_layers_list[pos].layer_output(self.hidden_layers_list[pos-1].a,
                                                          self.last_layer.beta,
                                                          self.last_layer.w,
                                                          self.last_layer.z)
            else:
                self.hidden_layers_list[pos].layer_output(self.hidden_layers_list[pos-1].a,
                                                          self.hidden_layers_list[pos+1].beta,
                                                          self.hidden_layers_list[pos+1].w,
                                                          self.hidden_layers_list[pos+1].z)

    def _validate(self, fun, samples, targets, n, converter):
        assert fun is not None
        mse = 0
        errn = 0
        for i in range(n):
            self.input_layer.layer_output(samples[i])
            for j in range(len(self.hidden_layers_list)):
                fun()
            self.last_layer.layer_output(self.hidden_layers_list[-1].a, targets[i])
            mse += auxiliaries.mean_squared_error(self.last_layer.z, targets[i])
            y = converter(targets[i])
            mx, index = auxiliaries.get_max_index(self.last_layer.z)
            if index != y:
                errn += 1
        print("\nMSE: %s" % str(mse/n))
        print("\nCORR: %s / %s" % (str(n - errn), n))

    def validate(self, samples, targets, n):
        assert n > 0
        assert len(samples) == len(targets) == n
        n_of_layers = len(self.hidden_layers_list)
        if n_of_layers == 1:
            self._validate(self._validate_single_hidden_layer, samples, targets,
                           n, auxiliaries.convert_binary_to_number)
        else:
            self._validate(self._validate_hidden_layers, samples, targets,
                           n, auxiliaries.convert_binary_to_number)


def main():
    fnn = FNN(768, 10, 100)
    c = 200
    samples, targets = auxiliaries.data_gen(768, 10, c)
    fnn.train(samples, targets, c)

    test = 100
    samples, targets = auxiliaries.data_gen(768, 10, test)
    fnn.validate(samples, targets, test)


if __name__ == "__main__":
    main()
