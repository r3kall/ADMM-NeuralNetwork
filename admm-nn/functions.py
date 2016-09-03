
import numpy as np

__author__ = 'Lorenzo Rutigliano, lnz.rutigliano@gmail.com'

##########################################################################################
##              Cost Functions ---> z, y are scalars                                    ##
##########################################################################################


def bhe(z, y):
    # Binary hinge error
    if y == 1:
        return np.maximum(0, 1 - z)
    return np.maximum(0, z)
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


def loglh(z, y):
    # Log Likehood
    err = np.seterr(all='ignore')
    score = - (z * np.log(y) + (1 - z) * np.log(1 - y))
    np.seterr(divide=err['divide'], over=err['over'],
              under=err['under'], invalid=err['invalid'])
    if np.isnan(score):
        score = 0
    return score
# end cost function


cost = {
    "binary_loss"           : bhe,
    "absolute_error"        : abse,
    "squarred_error"        : sqe,
    "squarred_log_error"    : sqloge,
    "log_likehood"          : loglh,
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
    "sigmoid"    : ndsigmoid,
}

##########################################################################################
##              Evaluation Functions ---> z, y are numpy matrix                         ##
##########################################################################################

# MEANS WRONG
def mbhe(z, y):
    # Mean binary hinge error
    return np.mean([abse(z[i, j], y[i, j]) for i, j in
                    zip(range(z.shape[0]), range(z.shape[1]))], dtype='float64')
# end evaluation function


def cle(z, y):
    # Classification error
    return np.sum([[1.0 for i, j in zip(z[:, w], y[:, w]) if i != j] for w in
                   range(z.shape[1])], dtype='float64') / (z.shape[0] * z.shape[1])
# end evaluation function


def mabse(z, y):
    # Mean absolute error
    return np.mean([abse(z[i, j], y[i, j]) for i, j in
                    zip(range(z.shape[0]), range(z.shape[1]))], dtype='float64')
# end evaluation function


def msqe(z, y):
    # Mean squarred error
    return np.mean([sqe(z[i, j], y[i, j]) for i, j in
                    zip(range(z.shape[0]), range(z.shape[1]))], dtype='float64')
# end evaluation function


def msqloge(z, y):
    # Mean squarred log error
    return np.mean([sqloge(z[i, j], y[i, j]) for i, j in
                    zip(range(z.shape[0]), range(z.shape[1]))], dtype='float64')
# end evaluation function


evaluation = {

}


