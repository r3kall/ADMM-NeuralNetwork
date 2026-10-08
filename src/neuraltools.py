import numpy as np


__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'


def generate_weights(t):
    return [np.zeros((t[i], t[i - 1]), dtype=np.float64) for i in range(1, len(t))]


def generate_outputs(t, s, rng=None):
    rng = np.random.default_rng(rng)
    return [rng.standard_normal((t[i], s)) for i in range(1, len(t))]


def generate_activations(t, s, rng=None):
    rng = np.random.default_rng(rng)
    return [rng.standard_normal((t[i], s)) for i in range(1, len(t) - 1)]


def _indices(instance, shuffle, rng):
    n = instance.samples.shape[1]
    return np.random.default_rng(rng).permutation(n) if shuffle else np.arange(n)


def get_sub_instance(instance, percentage=25, shuffle=False, rng=None):
    from .neuralnetwork import Instance
    from .commons import get_percentage
    n = get_percentage(percentage, instance.samples.shape[1])
    indices = _indices(instance, shuffle, rng)[:n]
    if n == 0:
        raise ValueError("subset must contain at least one sample")
    return Instance(instance.samples[:, indices], instance.targets[:, indices])


def split_instance(instance, percentage=25, shuffle=False, rng=None):
    """Return remainder, selected subset without changing the source instance."""
    from .neuralnetwork import Instance
    from .commons import get_percentage
    if not 0 < percentage < 100:
        raise ValueError("percentage must be between 0 and 100, exclusive")
    n = get_percentage(percentage, instance.samples.shape[1])
    if not 0 < n < instance.samples.shape[1]:
        raise ValueError("both partitions must contain samples")
    indices = _indices(instance, shuffle, rng)
    parts = (indices[n:], indices[:n])
    return tuple(Instance(instance.samples[:, ix], instance.targets[:, ix])
                 for ix in parts)


def save_network_to_file(net, filename="network0.pkl"):
    import pickle, os, re
    """
    This save method pickles the parameters of the current network into a
    binary file for persistent storage.
    """

    if filename == "network0.pkl":
        while os.path.exists(os.path.join(os.getcwd(), filename)):
            filename = re.sub(r'\d(?!\d)', lambda x: str(int(x.group(0)) + 1), filename)

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
    from .neuralnetwork import NeuralNetwork
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

    net = NeuralNetwork(training_space, features, classes,
                                          *layers, beta=beta, gamma=gamma)
    net.w = [np.asarray(w, dtype=np.float64) for w in weights]
    net.z = [np.asarray(z, dtype=np.float64) for z in outputs]
    net.a = [np.asarray(a, dtype=np.float64) for a in activations]
    net.l = np.asarray(l, dtype=np.float64)
    return net
# end tool

