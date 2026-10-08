import numpy as np

from .algorithms.admm import weight_update, activation_update, argminz, lambda_update
from .logger import defineLogger, Loggers, Levels
from .neuraltools import generate_weights, generate_outputs, generate_activations
log = defineLogger(Loggers.STANDARD)
log.setLevel(Levels.INFO.value)

__author__ = "Lorenzo Rutigliano, lnz.rutigliano@gmail.com"


class NeuralNetwork(object):
    # Neural Network Model
    def __init__(self, training_space, features, classes, *layers,
                 beta=1., gamma=10., code='binary', rng=None):
        """
        Create a neural network with random starting values

        :param training_space:  number of samples
        :param features:        number of features
        :param classes:         number of classes
        :param layers:          tuple of layers dimension
        :param beta:            double costant
        :param gamma:           double costant
        :param code:            set activation/loss function
        """

        sizes = (training_space, features, classes) + layers
        if not layers or any(isinstance(n, (bool, np.bool_)) or
                             not isinstance(n, (int, np.integer)) or n <= 0 for n in sizes):
            raise ValueError("sample count and all layer dimensions must be positive integers")
        if not np.isfinite(beta) or not np.isfinite(gamma) or beta <= 0 or gamma <= 0:
            raise ValueError("beta and gamma must be finite and positive")
        self.rng = np.random.default_rng(rng)
        self.parameters = (training_space, features, classes, layers)
        self.dim = len(layers) + 1

        self.beta = beta
        self.gamma = gamma

        self.argminlastz, self.activation_function, self.loss_fn = setalg(code)

        t = (features,) + layers + (classes,)
        self.w = generate_weights(t)
        self.z = generate_outputs(t, training_space, self.rng)
        self.a = generate_activations(t, training_space, self.rng)
        self.l = np.zeros((classes, training_space), dtype=np.float64)
    # end

    def train(self, training_data, training_targets):
        training_data, training_targets = self._validate_training(training_data, training_targets)
        self._train_hidden_layers(training_data)
        self.w[-1] = weight_update(self.z[-1], self.a[-1])
        self.z[-1] = self.argminlastz(training_targets, self.l, self.w[-1], self.a[-1], self.beta)
        self.l += lambda_update(self.z[-1], self.w[-1], self.a[-1], self.beta)
    # end

    def warmstart(self, training_data, training_targets):
        # Train the net without the Lagrangian update
        training_data, training_targets = self._validate_training(training_data, training_targets)
        self._train_hidden_layers(training_data)
        self.w[-1] = weight_update(self.z[-1], self.a[-1])
        self.z[-1] = self.argminlastz(training_targets, self.l, self.w[-1], self.a[-1], self.beta)
    # end

    def _validate_training(self, data, targets):
        data = self._validate_data(data)
        targets = validate_targets(targets)
        if data.shape[1] != self.parameters[0]:
            raise ValueError("training sample count differs from network state")
        if targets.shape != (self.parameters[2], data.shape[1]):
            raise ValueError("target shape must be (classes, samples)")
        return data, targets.astype(np.uint8, copy=False)

    def _validate_data(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] != self.parameters[1] or data.shape[1] == 0:
            raise ValueError("data shape must be (features, nonzero samples)")
        if not np.isfinite(data).all():
            raise ValueError("data must be finite")
        return data

    def _train_hidden_layers(self, a):
        self.w[0] = weight_update(self.z[0], a)
        self.a[0] = activation_update(self.w[1], self.z[1],
                                      self.activation_function(self.z[0]),
                                      self.beta, self.gamma)
        self.z[0] = argminz(self.a[0], self.w[0], a, self.gamma, self.beta)

        for i in range(1, self.dim - 1):
            self.w[i] = weight_update(self.z[i], self.a[i - 1])
            self.a[i] = activation_update(self.w[i + 1], self.z[i + 1],
                                          self.activation_function(self.z[i]),
                                          self.beta, self.gamma)
            self.z[i] = argminz(self.a[i], self.w[i], self.a[i - 1], self.gamma, self.beta)
    # end

    def feedforward(self, data):
        # This is a forward operation in the network. This is how we
        # calculate the network output from a set of input signals.
        data = self._validate_data(data)
        for i in range(self.dim - 1):
            data = self.activation_function(np.dot(self.w[i], data))
        # In the last layer we don't use the activation function
        return np.dot(self.w[-1], data)
    # end

    def error(self, data, targets):
        # perform a forward operation to calculate the output signal
        out = self.feedforward(data)
        # evaluate the output signal with the evaluation function
        targets = validate_targets(targets)
        if targets.shape != out.shape:
            raise ValueError("target shape must match network output")
        return self.loss_fn(out, targets)
    # end
# end class NeuralNetwork


def setalg(code):
    if code == 'binary':
        from .algorithms.hingebinary import argminlastz
        from .functions import relu, mbhe
        return argminlastz, relu, mbhe
    else:
        raise ValueError(f"unknown algorithm: {code!r}")
# end


class Instance(object):
    # This is a simple encapsulation of a `input signal : output signal`
    # pair in our training set.
    def __init__(self, samples, targets, intype=np.float64, outtype=np.uint8):
        self.samples = np.array(samples, dtype=intype, copy=True)
        targets = validate_targets(targets)
        if (self.samples.ndim != 2 or 0 in self.samples.shape
                or self.samples.shape[1] != targets.shape[1]):
            raise ValueError("samples and targets must be nonempty 2D arrays with equal sample counts")
        if not np.isfinite(self.samples).all():
            raise ValueError("samples must be finite")
        self.targets = np.array(targets, dtype=outtype, copy=True)


def validate_targets(targets):
    """Validate before dtype conversion so invalid labels cannot be truncated."""
    targets = np.asarray(targets)
    if targets.ndim != 2 or 0 in targets.shape:
        raise ValueError("targets must be a nonempty 2D array")
    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("targets must contain only 0 and 1")
    if not np.all(targets.sum(axis=0) == 1):
        raise ValueError("targets must be one-hot columns")
    return targets
