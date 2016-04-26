from keras import backend as K
from .core import Dense, Lambda, Reshape, Activation, Flatten
from .wrappers import Distribute, Wrapper
from .recurrent import LSTM
from .embeddings import Embedding
from ..engine import merge, Layer, InputSpec
from ..activations import softmax
import numpy as np


def get_shape(x):        
    if hasattr(x, '_keras_shape'):
        input_shape = x._keras_shape
    elif hasattr(K, 'int_shape'):
        input_shape = K.int_shape(x)
    else:
        raise Exception("I'm not sure why, but " + x.name + "doesn't have"
                        " shape information.. this is a bummer"
                        " and a problem")
    return input_shape

def make_safe(x):
    return K.clip(x, K.common._EPSILON, 1.0 - K.common._EPSILON)

class DenseFork(MaxoutDense):
    def __init__(self, output_dim, num_forks, *args, **kwargs):
        if 'nb_features' in kwargs:
            kwargs.pop('nb_features')
        super(DenseFork, self).__init__(output_dim, nb_features=num_forks, *args, **kwargs)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], input_shape[1], self.output_dim)

    def call(self, x, mask=None):
        # no activation, this layer is only linear.
        output = K.dot(x, self.W) + self.b
        return output


class ProbabilityTensor(Wrapper):
    """ function for turning 3d tensor to 2d probability matrix """
    def __init__(self, dense_function=None, *args, **kwargs):
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        layer = Distribute(dense_function) or Distribute(Dense(1, name='ptensor_func'))
        super(ProbabilityTensor, self).__init__(layer, *args, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.input_spec = [InputSpec(shape=input_shape)]
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis.')

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
            
        super(ProbabilityTensor, self).build()

    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,n 
        #       s.t. \sum_n n = 1
        return (input_shape[0], input_shape[1])

    def squash_mask(self, mask):
        if K.ndim(mask) == 2:
            return mask
        elif K.ndim(mask) == 3:
            return K.any(mask, axis=-1)

    def compute_mask(self, x, mask=None):
        return None

    def call(self, x, mask=None):
        if isinstance(x, list):
            x, mask = x[0], K.not_equal(x[1], 0)
        energy = K.squeeze(self.layer(x), 2)
        p_matrix = softmax(energy)
        if mask is not None:
            mask = self.squash_mask(mask)
            p_matrix = make_safe(p_matrix * mask)
            p_matrix = (p_matrix / K.sum(p_matrix, axis=-1, keepdims=True))*mask
        return p_matrix

    def get_config(self):
        config = {}
        base_config = super(ProbabilityTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class SoftAttention(ProbabilityTensor):
    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,f where f is weighted features summed across n
        return (input_shape[0], input_shape[2])

    def compute_mask(self, x, mask=None):
        if mask is None or mask.ndim==2:
            return None
        elif mask.ndim==3:
            mask = K.any(mask, axis=(1,2))
        else:
            raise Exception("Unexpected situation")

    def call(self, x, mask=None):
        # b,n,f -> b,f via b,n broadcasted
        p_vectors = K.expand_dims(super(SoftAttention, self).call(x, mask), 2)
        expanded_p = K.repeat_elements(p_vectors, K.shape(x)[2], axis=2)
        return K.sum(expanded_p * x, axis=1)

class Fix(Flatten):
    '''Flattens the input. Does not affect the batch size.

    # Example

    ```python
        model = Sequential()
        model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
        # now: model.output_shape == (None, 64, 32, 32)

        model.add(Flatten())
        # now: model.output_shape == (None, 65536)
    ```
    '''
    def __init__(self, return_mask=True, **kwargs):
        super(Fix, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_mask = return_mask

    def compute_mask(self, x, mask=None):
        if mask is None or not self.return_mask:
            return None
        return K.batch_flatten(mask)


class LambdaMask(Layer):
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.supports_masking = True
        super(LambdaMask, self).__init__(*args, **kwargs)

    def compute_mask(self, x, mask=None):
        return self.func(x, mask)

    def call(self, x, mask=None):
        return x

class Summarize(Wrapper):
    #def __init__(self, summary_space_size, *args, **kwargs):
    #    self.summary_space_size = summary_space_size
    def __init__(self, summarizer, *args, **kwargs):
        self.supports_masking = True
        super(Summarize, self).__init__(summarizer, *args, **kwargs)

    def build(self, input_shape=None):
        '''Assumes that self.layer is already set.
        Should be called at the end of .build() in the
        children classes.
        '''        
        assert len(input_shape) > 3
        self.input_spec = [InputSpec(shape=input_shape)]
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis.')



        child_input_shape = (np.prod(input_shape[:-2]),) + input_shape[-2:]
        if not self.layer.built:
            self.layer.build(child_input_shape)

        self.trainable_weights = getattr(self.layer, 'trainable_weights', [])
        self.non_trainable_weights = getattr(self.layer, 'non_trainable_weights', [])
        self.updates = getattr(self.layer, 'updates', [])
        self.regularizers = getattr(self.layer, 'regularizers', [])
        self.constraints = getattr(self.layer, 'constraints', {})

    def get_output_shape_for(self, input_shape):
        child_input_shape = (1,) + input_shape[-2:]
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        return input_shape[:-2] + child_output_shape[-1:]
        #return input_shape[:-2] + (self.summary_space_size,)

    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        target_dim = K.ndim(x) - 2
        num_reducing = K.ndim(mask) - target_dim
        if num_reducing:
            axes = tuple([-i for i in range(1,num_reducing+1)])
            mask = K.max(mask, axes)
        return mask

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        x = K.reshape(x, (-1,) + input_shape[-2:]) # (batch * d1 * ... * dn-2, dn-1, dn)
        mask_shape = (K.shape(x)[0], -1)
        mask = K.reshape(mask, mask_shape) # give it the same first dim
        y = self.layer.call(x, mask)
        output_shape = self.get_output_shape_for(input_shape)
        return K.reshape(y, output_shape)

    def get_config(self):
        config = {}  #'summary_space_size': self.summary_space_size
        base_config = super(Summarize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))