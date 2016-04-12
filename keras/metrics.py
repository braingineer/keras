from . import backend as K
from operator import mul

from .utils.generic_utils import get_from_module

def binary_accuracy(y_true, y_pred, mask=None):
    return K.mean(K.equal(y_true, K.round(y_pred)))


def categorical_accuracy(y_true, y_pred, mask=None):
    if mask is not None:
        eval_shape = (reduce(mul, y_true.shape[:-1]), y_true.shape[-1])
        y_true_ = K.reshape(y_true, eval_shape)
        y_pred_ = K.reshape(y_pred, eval_shape)
        flat_mask = K.flatten(mask)
        comped = K.equal(K.argmax(y_true_, axis=-1),
                          K.argmax(y_pred_, axis=-1))
        ## not sure how to do this in tensor flow
        good_entries = flat_mask.nonzero()[0]
        return K.mean(K.gather(comped, good_entries))

    else:
        return K.mean(K.equal(K.argmax(y_true, axis=-1),
                              K.argmax(y_pred, axis=-1)))



def perplexity(y_true, y_pred, mask=None):
    if mask is not None:
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        mask = K.permute_dimensions(K.reshape(mask, y_true.shape[:-1]), (0, 1, 'x'))
        truth_mask = K.flatten(y_true*mask).nonzero()[0]
        predictions = K.gather(y_pred.flatten(), truth_mask)
        return K.pow(2, K.mean(-K.log2(predictions)))
        ## not sure how to do this in tensor flow
        #out = K.switch(out > 0, -K.log2(out), 0)
        #return K.pow(2, K.mean(out))
    else:
        return K.pow(2, K.mean(-K.log2(K.equal(K.argmax(y_true, axis=-1),
                                               K.argmax(y_pred, axis=-1)))))







def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'metrics',
                           instantiate=False, kwargs=kwargs)
