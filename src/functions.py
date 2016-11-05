
import numpy as np

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'

##########################################################################################
##              Cost Functions ---> z, y are scalars                                    ##
##########################################################################################


def bhe(z, y):
    # Binary hinge error
    if y == 0:
        return np.maximum(0, z)
    else:
        return np.maximum(0, 1 - z)
# end cost function


def abse(z, y):
    # Absolute error
    return np.abs(z - y)
# end cost function


def sqe(z, y):
    # Squarred error
    return np.power(z - y, 2)
# end cost function


def sqloge(z, y):
    # Squarred Log error
    return np.power(np.log(z + 1) - np.log(y + 1), 2)
# end cost function


cost = {
    "binary_loss"           : bhe,
    "absolute_error"        : abse,
    "squarred_error"        : sqe,
    "squarred_log_error"    : sqloge
}

##########################################################################################
##              Activation Functions ---> signal is a scalar                            ##
##########################################################################################


def relu(signal):
    # ReLU function
    return np.maximum(0, signal)
# end activation function


def ndsigmoid(signal):
    # Non-differentiable sigmoid function
    return np.minimum(1, np.maximum(0, signal))
# end activation function


activation = {
    "relu"          : relu,
    "sigmoid"       : ndsigmoid
}

##########################################################################################
##              Evaluation Functions ---> z, y are numpy matrix                         ##
##########################################################################################


def mbhe(z, y):
    # Mean binary hinge error
    e = [bhe(z[k, w], y[k, w]) for k in range(z.shape[0]) for w in range(z.shape[1])]
    return np.mean(e, dtype='float64')
# end evaluation function


def mabse(z, y):
    # Mean absolute error
    e = [abse(z[k, w], y[k, w]) for k in range(z.shape[0]) for w in range(z.shape[1])]
    return np.mean(e, dtype='float64')
# end evaluation function


def msqe(z, y):
    # Mean squarred error
    e = [sqe(z[k, w], y[k, w]) for k in range(z.shape[0]) for w in range(z.shape[1])]
    return np.mean(e, dtype='float64')
# end evaluation function


def msqloge(z, y):
    # Mean squarred log error
    e = [sqloge(z[k, w], y[k, w]) for k in range(z.shape[0]) for w in range(z.shape[1])]
    return np.mean(e, dtype='float64')
# end evaluation function


evaluation = {
    "mean_binary_loss"          : mbhe,
    "mean_absolute_error"       : mabse,
    "mean_squarred_error"       : msqe,
    "mean_squarred_log_error"   : msqloge
}


