# General imports
import math
import numpy as np

# Keras imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.utils.generic_utils import get_custom_objects
import keras.backend as K

# Helper functions
import load_model_config
import keras_helpers

keras_helpers.define_custom_activations()

def create_model(in_size, out_size, path = "."):
    config = load_model_config.Config(path)

    np.random.seed(config.rand_seed)

    # Optimizer parameters
    if config.opt.name == 'sgd':
      optimizer = keras.optimizers.SGD(lr = config.lr, momentum = config.opt.momentum, nesterov = config.opt.nesterov)
    elif config.opt.name == 'adam':
      epsilon = config.opt.eps_coeff ** config.opt.eps_exp
      optimizer = keras.optimizers.Adam(lr = config.lr, beta_1 = config.opt.beta_1, beta_2 = config.opt.beta_2, epsilon = epsilon, amsgrad = config.opt.amsgrad)

    # Regularizers
    kernel_reg = keras.regularizers.L1L2(l1 = config.kernel_reg.l1, l2 = config.kernel_reg.l2)
    bias_reg = keras.regularizers.L1L2(l1 = config.bias_reg.l1, l2 = config.bias_reg.l2)

    # Hidden layer sizes
    # If n(i) -> n(i+1) is an EQL layer, then n(i) = u + 2v and n(i+1) = u + v.
    # Then, v = n(i) - n(i+1) and u = n(i+1) - v = 2n(i+1) - n(i).
    # Then, for both u > 0 and v > 0, we need n(i+1) < n(i) < 2n(i+1).
    units = (in_size, *[layer.in_units for layer in config.hidden[1:]], out_size)

    # Create the model
    model = Sequential()

    for i in range(len(units) - 1):
        if isinstance(config.hidden[i].act, str):
            # Initializer
            # mean = 0.0
            # stddev = 1 / math.sqrt(units[i])
            # initial = keras.initializers.RandomNormal(mean = mean, stddev = stddev)

            # Hidden layer
            model.add(Dense(units[i + 1], input_shape = (units[i],),
                                          activation = config.hidden[i].act,
                                          kernel_initializer = 'normal',#initial,
                                          bias_initializer = 'normal',#initial,
                                          kernel_regularizer = kernel_reg,
                                          bias_regularizer = bias_reg))
        elif isinstance(config.hidden[i].act, dict) and "eql" in config.hidden[i].act.keys():
            # EQL Layer
            # model.add(Lambda(keras_helpers.eql_activation_twofunc, arguments=kwd_2f1))
            eql_args = {'u': 2 * units[i + 1] - units[i], 'v': units[i] - units[i + 1], 'unary_act': config.hidden[i].act['eql']}
            model.add(Lambda(keras_helpers.eql_activation, arguments = eql_args))

        if config.hidden[i].norm:
            model.add(BatchNormalization())
        if config.hidden[i].dropout:
            model.add(Dropout(config.dropout))

    # Configure the model for training
    model.compile(optimizer = optimizer, loss = config.loss, metrics = config.metrics)

    return model