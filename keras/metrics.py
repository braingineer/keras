from . import backend as K
from operator import mul

from .utils.generic_utils import get_from_module

def binary_accuracy(y_true, y_pred, mask=None):
    return K.mean(K.equal(y_true, K.round(y_pred)))


def categorical_accuracy(y_true, y_pred, mask=None):
    if mask is not None:
        if K.ndim(y_pred) == K.ndim(mask):
            mask = K.max(mask, axis=-1)
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







def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'metrics',
                           instantiate=False, kwargs=kwargs)
