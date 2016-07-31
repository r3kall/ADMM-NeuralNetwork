import auxiliaries
from model.layers import InputLayer, HiddenLayer, LastLayer
import numpy as np

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


class FNN(object):
    def __init__(self, data, sample_length, number_of_classes, *hidden_layers):
        # if data is None raise
        # if hidden_layers empty raise
        self.data = data
        self.sample_length = sample_length
        self.number_of_classes = number_of_classes
        self.hidden_layers_list = []

        for index, dim in enumerate(hidden_layers):
            if index == 0:
                n_in = self.sample_length
            else:
                n_in = hidden_layers[index - 1]

            self.hidden_layers_list.append(HiddenLayer(n_in, dim))

        self.input_layer = InputLayer(self.sample_length, self.sample_length, data)
        self.last_layer = LastLayer(self.hidden_layers_list[-1].n_out,
                                    self.number_of_classes, data)

    def __str__(self):
        hl = ''
        for item in self.hidden_layers_list:
            hl = hl + item.__str__() + ' -> '
        return self.input_layer.__str__() + ' -> ' + hl + self.last_layer.__str__()


def main():
    fnn = FNN(None, 800, 10, 100)
    #print(fnn.__str__())
    #a = np.arange(10).reshape(2, 5)
    #auxiliaries.check_dimensions(a, 2, 5)
    pass



if __name__ == "__main__":
    main()
