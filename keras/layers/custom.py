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

class ProbabilityTensor(Layer):
    """ function for turning 3d tensor to 2d probability matrix """
    def __init__(self, dense_function=None, *args, **kwargs):
        self.input_spec = [InputSpec(ndim=3)]
        self.p_func = dense_function or Dense(1)
        self.supports_masking = True
        super(ProbabilityTensor, self).__init__(*args, **kwargs)

    def get_output_shape_for(self, input_shape):
        # b,n,f -> b,n 
        #       s.t. \sum_n n = 1
        if isinstance(input_shape, (list,tuple)) and not isinstance(input_shape[0], int):
            input_shape = input_shape[0]

        return (input_shape[0], input_shape[1])

    def squash_mask(self, mask):
        if K.ndim(mask) == 2:
            return mask
        elif K.ndim(mask) == 3:
            mask = K.max(mask, axis=-1)
        elif K.ndim(mask) > 3:
            raise Exception("what?")
        return mask

    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        if isinstance(x, list):
            return K.not_equal(x[1], 0)
        return self.squash_mask(mask)

    def call(self, x, mask=None):
        if isinstance(x, list):
            x, mask = x[0], K.not_equal(x[1], 0)
        energy = K.squeeze(Distribute(self.p_func)(x), 2)
        p_matrix = softmax(energy)
        if mask is not None:
            mask = self.squash_mask(mask)
            p_matrix = make_safe(p_matrix * mask)
            p_matrix = (p_matrix / K.sum(p_matrix, axis=-1, keepdims=True))*mask
        if p_matrix.ndim > 2:
            raise Exception("fuck")
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
        if mask is None:
            return None
        if mask.ndim==3:
            mask = K.max(mask, axis=(1,2))
        elif mask.ndim==2:
            mask = K.max(mask, axis=(1,))
        else:
            raise Exception("Unexpected situation")
        mask = K.expand_dims(mask, -1)
        return mask

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
    def __init__(self, **kwargs):
        super(Fix, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        return K.batch_flatten(mask)


class Mergecast(Layer):
    def __init__(self, output_shape, input_shapes, *args, **kwargs):
        self.output_size = output_shape
        super(Mergecast, self).__init__(*args, **kwargs)
        self.input_spec = [InputSpec(shape=in_shape) for in_shape in input_shapes]

    def get_output_shape_for(self, input_shapes):
        base = input_shapes[0][:2]
        last = sum([x[-1] for x in input_shapes])
        out = base + (last,)
        return out

    def compute_mask(self, x, mask=None):
        return None
        if mask is None:
            return None
        import pdb
        #pdb.set_trace()
        out = None
        for m in mask:
            #m = K.expand_dims(K.max(m, axis=-1), -1)
            if out is None:
                out = m
            else:
                out = out * m
        return out

    def call(self, xs, mask=None):
        #input_shapes = [spec.shape for spec in self.input_spec]
        return K.concatenate(xs, axis=-1)

    def get_config(self):
        config = {}
        base_config = super(Mergecast, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))   


class MultiEmbedding(Layer):
    """ function for embedding with encoding """
    def __init__(self,  bit_info, mask_zero=False, *args, **kwargs):
        # bit info => (in_dim, out_dim) for each item
        ## assert sameness
        self.output_size = sum([O for I,O in bit_info])
        self.bit_info = bit_info
        self.mask_zero = mask_zero
        self.layers = []
        for I, O in bit_info:
            self.layers.append(Embedding(I, O, mask_zero=mask_zero))

        super(MultiEmbedding, self).__init__(*args, **kwargs)

    def build(self, input_shapes):
        '''Assumes that self.layer is already set.
        Should be called at the end of .build() in the
        children classes.
        '''
        self.trainable_weights = []
        self.non_trainable_weights = [] 
        self.updates = []
        self.regularizers =  []
        self.constraints = {}
        self.input_spec = [InputSpec(shape=I) for I in input_shapes]
        for layer, shape in zip(self.layers, input_shapes):
            layer.build(shape)

        for layer in self.layers:
            self.trainable_weights +=  getattr(layer, 'trainable_weights', [])
            self.non_trainable_weights += getattr(layer, 'non_trainable_weights', [])
            self.updates  += getattr(layer, 'updates', [])
            self.regularizers  += getattr(layer, 'regularizers', [])
            self.constraints.update(getattr(layer, 'constraints', {}))

    #def get_weights(self):
    #    weights = [weight for layer in self.layers for weight in layer.get_weights()]
    #    return weights

    #def set_weights(self, weights):
    #    self.layer.set_weights(weights)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0] + (self.output_size,)

    def call(self, x, mask=None):
        inputs = [F_emb.call(x_i) for F_emb, x_i in zip(self.layers, x)]
        return K.concatenate(inputs, axis=-1)
        
    def compute_mask(self, x, mask=None):
        if self.mask_zero is False:
            return None
        else:
            return K.not_equal(K.expand_dims(x[0]), 0)
            #for x_i in x[1:]:
            #    out.append(K.not_equal(K.expand_dims(x_i), 0))
            #return K.concatenate([K.not_equal(x_i, 0) for x_i in x], axis=-1)
    
    def get_config(self):
        config = {'output_size': self.output_size}
        base_config = super(MultiEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class CustomReshape(Reshape):
    def __init__(self, *args, **kwargs):
        super(CustomReshape, self).__init__(*args, **kwargs)
        self.supports_masking = True

    def get_output_shape_for(self, input_shape):
        return self._fix_unknown_dimension(input_shape, self.target_shape)

    def get_shape(self, x):
        # In case the target shape is not fully defined,
        # we need access to the shape of x.
        # solution:
        # 1) rely on x._keras_shape
        # 2) fallback: K.int_shape
        target_shape = self.target_shape
        if -1 in target_shape:
            # target shape not fully defined
            input_shape = None
            if hasattr(x, '_keras_shape'):
                input_shape = x._keras_shape
            elif hasattr(K, 'int_shape'):
                input_shape = K.int_shape(x)
            if input_shape is not None:
                target_shape = self.get_output_shape_for(input_shape)
        return target_shape

    def call(self, x, mask=None):
        #target_shape = get_shape(x)
        return K.reshape(x, self.target_shape)

    def compute_mask(self, x, mask=None):
        if mask is None:
            return mask
        else:
            mask_shape = self.target_shape[:-1] + (1,)
            return K.reshape(mask, mask_shape)

class Summarize(Layer):
    #def __init__(self, summary_space_size, *args, **kwargs):
    #    self.summary_space_size = summary_space_size
    def __init__(self, summarizer, *args, **kwargs):
        super(Summarize, self).__init__(*args, **kwargs)
        #self.F_rnn = LSTM(summary_space_size, name=self.name+"_rnn_func", unroll=unroll)
        self.supports_masking = True
        #self.return_mask = return_mask
        #self.layer = LSTM(summary_space_size, unroll=unroll)
        self.layer = summarizer


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

"""
def Summarize(space_size):
    def func(layer):
        sh = K.shape(layer)
        out_sh = 
        return Reshape(out_sh)(LSTM(space_size)(Reshape(rnn_sh)(layer)))
    return func
class Summarize(Lambda):
    \""" run an LSTM over the last 2 layers \"""
    def __init__(self, space_size, *args, **kwargs):
        self.space_size = space_size
        f = self.make() 
        import pdb
        pdb.set_trace()
        super(Summarize, self).__init__(self, f, *args, **kwargs)

    def make(self):
        def func(layer):
            sh = K.shape(layer)
            rnn_sh = (np.prod(sh[:-2]), ) + sh[-2:]
            out_sh = sh[:-2] + (self.space_size, )
            return Reshape(out_sh)(LSTM(self.space_size)(Reshape(rnn_sh)(layer)))
        return func

    def get_output_shape_for(self, input_shape):
        if isinstance(input_shape, (tuple, list)):
            raise Exception("Not prepared for this right now")

        return input_shape[:-2] + (self.space_size,)

    def compress_expand(self, shape):
        head, tail = shape[:-2], shape[-2:]
        c_shape = (np.prod(head),) + tail
        e_shape = head + (self.summary_space_size,)
        return (CustomReshape(c_shape, name=self.name+"_compress"), 
                CustomReshape(e_shape, name=self.name+"_expand"))
        #return Reshape(c_shape), Reshape(e_shape)
        #return Reshape((-1,)+shape[-2:]), Reshape((-1,self.summary_space_size))



        input_shape = get_shape(x) 
        c_shape = (np.prod(input_shape[:-2]),) + input_shape[-2:]
        e_shape = input_shape[:-2] + (self.summary_space_size, )
        F_compress, F_expand = self.compress_expand(input_shape)
        if mask is not None:
            mask = K.reshape(mask, c_shape[:-1]+(1,))
        #return K.reshape(self.F_rnn(K.reshape(x, c_shape), mask), e_shape)
        out = F_compress(x)
        out = self.F_rnn(out)
        out = F_expand(out)
        return out
        
"""