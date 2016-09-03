import numpy as np
import numpy.matlib

import neuralnetwork

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


def generate_weights(t):
    return [np.mat(np.zeros((t[i], t[i - 1]), dtype='float64')) for i in range(1, len(t))]
# end tool


def generate_gaussian(t, s):
    return [np.matlib.randn(t[i], s) for i in range(1, len(t))]
# end tool


def save_network_to_file(net, filename="network0.pkl"):
    import pickle, os, re
    """
    This save method pickles the parameters of the current network into a
    binary file for persistent storage.
    """

    if filename == "network0.pkl":
        while os.path.exists(os.path.join(os.getcwd(), filename)):
            filename = re.sub('\d(?!\d)', lambda x: str(int(x.group(0)) + 1), filename)

    with open(filename, 'wb') as file:
        store_dict = {
            "training_space"    : net.parameters[0],
            "features"          : net.parameters[1],
            "classes"           : net.parameters[2],
            "layers"            : net.parameters[3],

            "beta"              : net.beta,
            "gamma"             : net.gamma,

            "lambda"            : net.l,
            "weights"           : net.w,
            "outputs"           : net.z,
            "activations"       : net.a
        }
        pickle.dump(store_dict, file, -1)
# end tool


def load_network_from_file(filename):
    import pickle
    """
    Load the complete configuration of a previously stored network.
    """
    with open(filename, 'rb') as file:
        store_dict      = pickle.load(file)
        training_space  = store_dict["training_space"]
        features        = store_dict["features"]
        classes         = store_dict["classes"]
        layers          = store_dict["layers"]

        beta            = store_dict["beta"]
        gamma           = store_dict["gamma"]

        l               = store_dict["lambda"]
        weights         = store_dict["weights"]
        outputs         = store_dict["outputs"]
        activations     = store_dict["activations"]

    net = neuralnetwork.NeuralNetwork(training_space, features, classes,
                                      *layers, beta=beta, gamma=gamma)
    net.w = weights
    net.z = outputs
    net.a = activations
    net.l = l
    return net
# end tool

