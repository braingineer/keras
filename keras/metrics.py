import numpy as np
from . import backend as K
from operator import mul
from .utils.generic_utils import get_from_module
from functools import reduce

def normalize_mask(x, mask):
    '''Keep the mask align wtih the tensor x

    Arguments: x is a data tensor; mask is a binary tensor
    Rationale: keep mask at same dimensionality as x, but only with a length-1 
               trailing dimension. This ensures broadcastability, which is important
               because inferring shapes is hard and shapes are easy to get wrong. 
    '''
    mask = K.cast(mask, K.floatx())
    while K.ndim(mask) != K.ndim(x):
        if K.ndim(mask) > K.ndim(x):
            mask = K.any(mask, axis=-1)
        elif K.ndim(mask) < K.ndim(x):
            mask = K.expand_dims(mask)
    return K.any(mask, axis=-1, keepdims=True)

def binary_accuracy(y_true, y_pred, mask=None):
    return K.mean(K.equal(y_true, K.round(y_pred)))


def categorical_accuracy(y_true, y_pred, mask=None):
    if mask is not None:
        mask = normalize_mask(y_pred, mask)
        eval_shape = (reduce(mul, K.shape(y_true)[:-1]), K.shape(y_true)[-1])
        y_true_flat = K.reshape(y_true, eval_shape)
        y_pred_flat = K.reshape(y_pred, eval_shape)
        mask_flat = K.flatten(mask).nonzero()[0]  ### how do you do this on tensorflow?
        evalled = K.gather(K.equal(K.argmax(y_true_flat, axis=-1),
                                   K.argmax(y_pred_flat, axis=-1)), 
                           mask_flat)
        return K.mean(evalled)

    else:
        return K.mean(K.equal(K.argmax(y_true, axis=-1),
                              K.argmax(y_pred, axis=-1)))

def perplexity(y_true, y_pred, mask=None):
    if mask is not None:
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        mask = K.permute_dimensions(K.reshape(mask, y_true.shape[:-1]), (0, 1, 'x'))
        truth_mask = K.flatten(y_true*mask).nonzero()[0]  ### How do you do this on tensorflow?
        predictions = K.gather(y_pred.flatten(), truth_mask)
        return K.pow(2, K.mean(-K.log2(predictions)))
    else:
        return K.pow(2, K.mean(-K.log2(y_pred)))



def sparse_categorical_accuracy(y_true, y_pred):
    return K.mean(K.equal(K.max(y_true, axis=-1),
                          K.cast(K.argmax(y_pred, axis=-1), K.floatx())))


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf))
    return 100. * K.mean(diff)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), np.inf) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), np.inf) + 1.)
    return K.mean(K.square(first_log - second_log))


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)))


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.))


def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    return K.mean(K.categorical_crossentropy(y_pred, y_true))


def sparse_categorical_crossentropy(y_true, y_pred):
    '''expects an array of integer classes.
    Note: labels shape must have the same number of dimensions as output shape.
    If you get a shape error, add a length-1 dimension to labels.
    '''
    return K.mean(K.sparse_categorical_crossentropy(y_pred, y_true))


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true))


def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()))


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred)


# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity





def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'metrics',
                           instantiate=False, kwargs=kwargs)
