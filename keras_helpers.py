import numpy as np
from scipy import exp

# Keras imports
import keras
from keras.utils.generic_utils import get_custom_objects
import keras.backend as K

##########################################################
####              Activation functions                 ###
##########################################################

def linear_activation(x):
    return x

def log_activation(x):
    return K.log(x)

def exp_activation(x):
    return K.exp(x)

def sigmoid_activation(x):
    return K.sigmoid(x)

def square_activation(x):
    return K.square(x)

def power_activation(x,n):
    return K.pow(x,n)

def opf_activation(x):
    return power_activation(x, 1.5)

def cubic_activation(x):
    return power_activation(x, 3)

def eql_activation_old(x, u = 1, v = 1, unary_act = sigmoid_activation):
    """
    The activation function for the nonlinear part of the EQL (i.e. from z -> y), which will be used in a keras Lambda layer
    Note that we can pass a dictionary of keyword arguments to the Lambda layer constructor
    """

    # Separate the data into the unary and binary inputs
    unary_input = x[0:u]
    binary_input = x[u:]
    # binary_input = K.constant(K.eval(binary_input), shape=(2*v,))

    # Apply the unary activation function to the unary input
    unary_output = unary_act(unary_input)

    # Now perform the multiplication on the binary input
    # binary_input = K.eval(binary_input)
    binary_results = []
    for i in range(v):
        binary_results.append(binary_input[2 * i] * binary_input[2 * i + 1])
    binary_output = K.variable(binary_results)

    print("=====")
    print(unary_output.shape)
    print(binary_output.shape)

    # Concatenate the tensors and return the result
    return K.concatenate([unary_output, binary_output])


def eql_activation(x, u = 1, v = 1, unary_act = sigmoid_activation):
    """
    The activation function for the nonlinear part of the EQL (i.e. from z -> y), which will be used in a keras Lambda layer
    Note that we can pass a dictionary of keyword arguments to the Lambda layer constructor
    """

    # Separate the data into the unary and binary inputs
    unary_input = x[:, 0:u]
    binary_input1 = x[:, u:u + v]
    binary_input2 = x[:, u + v:u + 2 * v]
    unary_output = unary_act(unary_input)
    binary_output = keras.layers.multiply([binary_input1, binary_input2])
    print(unary_output.shape)
    print(binary_output.shape)
    return K.concatenate([unary_output, binary_output], axis = 1)



def eql_activation_twofunc(x, u = 1, v = 1, nu1 = 1, unary_act1 = sigmoid_activation, unary_act2 = sigmoid_activation):
    """
    The activation function for the nonlinear part of the EQL (i.e. from z -> y), which will be used in a keras Lambda layer
    Note that we can pass a dictionary of keyword arguments to the Lambda layer constructor
    """

    # Separate the data into the unary and binary inputs
    unary_input1 = x[:, 0:nu1]
    unary_input2 = x[:, nu1:u]
    binary_input1 = x[:, u:u + v]
    binary_input2 = x[:, u + v:u + 2 * v]
    # binary_input = K.constant(K.eval(binary_input), shape=(2*v,))

    # Apply the unary activation functions to the unary input
    unary_out1 = unary_act1(unary_input1)
    unary_out2 = unary_act2(unary_input2)
    unary_output = K.concatenate([unary_out1, unary_out2], axis = 1)

    # Now perform the multiplication on the binary input
    binary_output = keras.layers.multiply([binary_input1, binary_input2])

    # Concatenate the tensors and return the result
    print(unary_output.shape)
    print(binary_output.shape)
    return K.concatenate([unary_output, binary_output], axis = 1)



def define_custom_activations():
    """Calling this function allows Keras to use the custom activation functions by name."""
    get_custom_objects().update({"lin_act": linear_activation})
    get_custom_objects().update({"log_act": log_activation})
    get_custom_objects().update({"exp_act": exp_activation})
    get_custom_objects().update({"square_act": square_activation})
    get_custom_objects().update({"cubic_act": cubic_activation})
    get_custom_objects().update({"sigmoid_act": sigmoid_activation})
    get_custom_objects().update({"eql_act": eql_activation})



##########################################################
####                Results closeness                  ###
##########################################################

def within_n_percent(known, predicted, n = 10):
    """Returns True if predicted is within n% of known, False otherwise."""
    if predicted == known:
        return True
    if known == 0:
        return False
    r = float(n) / 100
    return(float(abs(known - predicted)) / float(known) < r)


def within_factor_of_n(known_val, pred_val, n):
    f = float(abs(known_val - pred_val) / float(known_val))
    if (f < n) and (f > 1 / n):
        return True
    return False


##########################################################
####                Scoring metrics                    ###
##########################################################

# def MAPE(actual, predicted):
#     """
#     Computes the MAPE. Ignores data with actual value == 0 to avoid divide-by-zero errors.
#     """
#     relerrs = []
#     for (x,y) in zip(actual, predicted):
#         if x == 0:
#             continue
#         else:
#             relerrs.append(abs((float(x) - float(y)) / float(x)))
#     N = len(relerrs)
#     hundredoverN = float(100) / float(N)
#     return hundredoverN * sum(relerrs)

# def MSE(actual, predicted):
#     N = len(actual)
#     sqerrs = [(float(x) - float(y)) ** 2 for (x, y) in zip(list(actual), list(predicted))]
#     return sum(sqerrs) / float(N)

def mean_absolute_percentage_error(y_true, y_pred):
    mask = (y_true != 0)
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

##########################################################
####              Learning rate schedulers             ###
##########################################################

def lr_drop(epoch, model, crit_epochs, factors):
    if epoch in crit_epochs:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr / float(factors[crit_epochs.index(epoch)]))
    return K.get_value(model.optimizer.lr)


##########################################################
####                      For fitting                  ###
##########################################################

# For the Gaussian fits
def gaus(x, a, x0, sigma):
    return a * exp(-((x - x0) ** 2) / (2 * sigma ** 2))


##########################################################
####                Miscellaneous                      ###
##########################################################

def model_name(model):
    return str(model.__class__).split('(')[0][:-2].split('.')[-1]

def flatten_doc(doc, prev_key = ""):
    new_doc = {}
    for key, value in doc.items():
        if isinstance(value, dict):
            new_doc.update(flatten_doc(value, prev_key = key))
        else:
            new_doc[prev_key + key] = value
    return new_doc


def is_pow(x,n):
    return int(math.log(x, n)) == float(math.log(x, n))