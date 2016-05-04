from keras import backend as K
from .core import Dense, Lambda, Reshape, Activation, Flatten, MaxoutDense
from .wrappers import Distribute, Wrapper
from .recurrent import LSTM
from .embeddings import Embedding
from ..engine import merge, Layer, InputSpec
from ..activations import softmax
import numpy as np
from ..engine.topology import Node

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


def _fix_unknown_dimension(input_shape, output_shape):
    '''Find and replace a single missing dimension in an output shape
    given an input shape.

    A near direct port of the internal numpy function _fix_unknown_dimension
    in numpy/core/src/multiarray/shape.c

    # Arguments
        input_shape: shape of array being reshaped

        output_shape: desired shape of the array with at most
            a single -1 which indicates a dimension that should be
            derived from the input shape.

    # Returns
        The new output shape with a -1 replaced with its computed value.

        Raises a ValueError if the total array size of the output_shape is
        different then the input_shape, or more then one unknown dimension
        is specified.
    '''
    output_shape = list(output_shape)

    msg = 'total size of new array must be unchanged'

    known, unknown = 1, None
    for index, dim in enumerate(output_shape):
        if dim < 0:
            if unknown is None:
                unknown = index
            else:
                raise ValueError('can only specify one unknown dimension')
        else:
            known *= dim

    original = np.prod(input_shape, dtype=int)
    if unknown is not None:
        if known == 0 or original % known != 0:
            raise ValueError(msg)
        output_shape[unknown] = original // known
    elif original != known:
        raise ValueError(msg)

    return tuple(output_shape)

class DenseFork(MaxoutDense):
    def __init__(self, output_dim, num_forks, *args, **kwargs):
        if 'nb_features' in kwargs:
            kwargs.pop('nb_features')
        super(DenseFork, self).__init__(output_dim, nb_feature=num_forks, *args, **kwargs)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], input_shape[1], self.output_dim)

    def call(self, x, mask=None):
        # no activation, this layer is only linear.
        output = K.dot(x, self.W) + self.b
        return output


class DynamicEmbedding(Embedding):
    def __init__(self, embedding_matrix, mode='matrix', *args, **kwargs):
        assert hasattr(embedding_matrix, '_keras_shape')
        self.W = embedding_matrix
        if mode=='tensor':
            assert len(embedding_matrix._keras_shape) == 3
            indim = self.W._keras_shape[1]
            outdim = self.W._keras_shape[2]
        else:
            assert len(embedding_matrix._keras_shape) == 2
            indim, outdim = self.W._keras_shape

        self.mode = mode
        super(DynamicEmbedding, self).__init__(indim, outdim, *args, **kwargs)
        
    def build(self, input_shape):
        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            
    def call(self, x, mask=None):
        if 0. < self.dropout < 1.:
            retain_p = 1. - self.dropout
            dims = self.W._keras_shape[:-1]
            B = K.random_binomial(dims, p=retain_p) * (1. / retain_p)
            B = K.expand_dims(B)
            W = K.in_train_phase(self.W * B, self.W)
        else:
            W = self.W
        
        if self.mode == 'matrix':
            return K.gather(W,x)
        elif self.mode == 'tensor':
            # quick and dirty: only allowing for 3dim inputs when it's tensor mode
            assert K.ndim(x) == 3
            # put sequence on first; gather; take diagonal across shared batch dimension
            # in other words, W is (B, S, F)
            # incoming x is (B, S, A)
            inds = K.arange(self.W._keras_shape[0])
            #out = K.gather(K.permute_dimensions(W, (1,0,2)), x).diagonal(axis1=0, axis2=3)
            #return K.permute_dimensions(out, (3,0,1,2))
            ### method above doesn't do grads =.=
            out = K.gather(K.permute_dimensions(W, (1,0,2)), x)
            out = K.permute_dimensions(out, (0,3,1,2,4))
            out = K.gather(out, (inds, inds))
            return out
        else:
            raise Exception('sanity check. should not be here.')

        #all_dims = T.arange(len(self.W._keras_shape))
        #first_shuffle = [all_dims[self.embed_dim]] + all_dims[:self.embed_dim] + all_dims[self.embed_dim+1:]
        ## 1. take diagonal from 0th to
        ## chang eof tactics
        ## embed on time or embed on batch. that's all I'm supporting.  
        ## if it's embed on time, then, x.ndim+1 is where batch will be, and is what
        ## i need to take the diagonal over. 
        ## now dim shuffle the xdims + 1 to the front.
        #todo: get second shuffle or maybe find diagonal calculations
        #out = K.gather(W, x)
        #return out

        ### reference
        #A = S(np.arange(60).reshape(3,4,5))
        #x = S(np.random.randint(0, 4, (3,4,10)))
        #x_emb = A.dimshuffle(1,0,2)[x].dimshuffle(0,3,1,2,4)[T.arange(A.shape[0]), T.arange(A.shape[0])]


class MutualEnergy(Activation):
    ''' assumes two inputs; assumes both are same shape '''
    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:-1]

    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        out = None
        for mask_i in mask:
            if mask_i is None:
                continue
            if out is None:
                out = mask_i
            else:
                out *= mask_i
        return out

    def call(self, xs, mask=None):
        x1, x2 = xs
        energy = x1*x2
        if mask is not None:
            for mask_i in mask:
                if mask_i is None: continue
                if K.ndim(mask_i) > K.ndim(x1):
                    mask_i = K.any(mask_i)
                elif K.ndim(mask_i) < K.ndim(x1):
                    mask_i = K.expand_dims(mask_i)
                energy *= mask_i
        energy = K.sum(energy, axis=-1)

        return self.activation(energy)



class ProbabilityTensor(Wrapper):
    """ function for turning 3d tensor to 2d probability matrix """
    def __init__(self, dense_function=None, *args, **kwargs):
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        if dense_function is None:
            dense_function = Dense(1, name='ptensor_func')
        layer = Distribute(dense_function)
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

class EZAttend(Layer):
    def __init__(self, p_tensor, *args, **kwargs):
        self.supports_masking = True
        self.p_tensor = p_tensor
        super(EZAttend, self).__init__(*args, **kwargs)

    def compute_mask(self, x, mask=None):
        return None

    def get_output_shape_for(self, input_shape):
        last_dim = K.ndim(self.p_tensor)
        output_shape = list(input_shape)
        output_shape.pop(last_dim-1)
        return tuple(output_shape)

    def call(self, target_tensor, mask=None):
        last_dim = K.ndim(self.p_tensor)
        expanded_p = K.repeat_elements(K.expand_dims(self.p_tensor, last_dim), 
                                       K.shape(target_tensor)[last_dim], 
                                       axis=last_dim)
        return K.sum(expanded_p * target_tensor, axis=last_dim-1)



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
    def __init__(self, summarizer, *args, **kwargs):
        self.supports_masking = True
        self.last_two = None
        super(Summarize, self).__init__(summarizer, *args, **kwargs)

    def build(self, input_shape=None):
        '''Assumes that self.layer is already set.
        Should be called at the end of .build() in the
        children classes.
        '''        
        ndim = len(input_shape)
        assert ndim >= 3
        self.input_spec = [InputSpec(ndim=str(ndim)+'+')]
        #if input_shape is not None:
        #    self.last_two = input_shape[-2:]
        self._input_shape = input_shape
        #self.input_spec = [InputSpec(shape=input_shape)]
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



        #child_input_shape = (np.prod(input_shape[:-2]),) + input_shape[-2:]
        child_input_shape = (None,)+input_shape[-2:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True

        self.trainable_weights = getattr(self.layer, 'trainable_weights', [])
        self.non_trainable_weights = getattr(self.layer, 'non_trainable_weights', [])
        self.updates = getattr(self.layer, 'updates', [])
        self.regularizers = getattr(self.layer, 'regularizers', [])
        self.constraints = getattr(self.layer, 'constraints', {})

    def get_output_shape_for(self, input_shape):
        child_input_shape = (1,) + input_shape[-2:]
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        return input_shape[:-2] + child_output_shape[-1:]

    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        #import pdb
        #pdb.set_trace()
        target_dim = K.ndim(x) - 2
        num_reducing = K.ndim(mask) - target_dim
        if num_reducing:
            axes = tuple([-i for i in range(1,num_reducing+1)])
            mask = K.any(mask, axes)

        return mask

    def call(self, x, mask=None):
        if hasattr(x, '_keras_shape'):
            input_shape = x._keras_shape
        else:
            input_shape = self._input_shape
        #import pdb
        #pdb.set_trace()
        #if self.last_two is not None:
        #    last2 = self.last_two
        #else:
        #    input_shape = x._keras_shape
        #    last2 = input_shape[-2:]
        #out_shape = K.shape(x)[:-2]

        x = K.reshape(x, (-1,) + input_shape[-2:]) # (batch * d1 * ... * dn-2, dn-1, dn)
        if mask is not None:
            mask_shape = (K.shape(x)[0], -1)
            mask = K.reshape(mask, mask_shape) # give it the same first dim
        y = self.layer.call(x, mask)
        #try:
        #output_shape = self.get_output_shape_for(K.shape(x))
        #except:
        output_shape =  self.get_output_shape_for(input_shape)
        #import pdb
        #pdb.set_trace()
        return K.cast(K.reshape(y, output_shape), K.floatx()) 

    def get_config(self):
        config = {}  #'summary_space_size': self.summary_space_size
        base_config = super(Summarize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class LastDimDistribute(Wrapper):
    def __init__(self, distributee, *args, **kwargs):
        self.supports_masking = True
        super(LastDimDistribute, self).__init__(distributee, *args, **kwargs)

    def build(self, input_shape=None):
        '''Assumes that self.layer is already set.
        Should be called at the end of .build() in the
        children classes.
        '''        
        assert len(input_shape) >= 3
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



        child_input_shape = (np.prod(input_shape[:-1]),) + input_shape[-1:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True

        self.trainable_weights = getattr(self.layer, 'trainable_weights', [])
        self.non_trainable_weights = getattr(self.layer, 'non_trainable_weights', [])
        self.updates = getattr(self.layer, 'updates', [])
        self.regularizers = getattr(self.layer, 'regularizers', [])
        self.constraints = getattr(self.layer, 'constraints', {})

    def get_output_shape_for(self, input_shape):
        child_input_shape = (1,) + input_shape[-1:]
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        return input_shape[:-1] + child_output_shape[-1:]

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        x = K.reshape(x, (-1,) + input_shape[-1:]) # (batch * d1 * ... * dn-2*dn-1, dn)
        mask_shape = (K.shape(x)[0], -1)
        mask = K.reshape(mask, mask_shape) # give it the same first dim
        y = self.layer.call(x, mask)
        output_shape = self.get_output_shape_for(input_shape)
        return K.reshape(y, output_shape)

    def get_config(self):
        config = {}  #'summary_space_size': self.summary_space_size
        base_config = super(LastDimDistribute, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class StackLSTM(LSTM):
    def build(self, input_shapes):
        assert isinstance(input_shapes, list)
        rnn_shape, indices_shape = input_shapes
        super(StackLSTM, self).build(rnn_shape)
        self.input_spec += [InputSpec(shape=indices_shape)]

    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.permute_dimensions(x, [1,0,2]) # (timesteps, samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (timesteps, samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[1], input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[1], input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[1], input_shape[0], self.output_dim)),
                           K.zeros((input_shape[1], input_shape[0], self.output_dim))]
    
    def get_output_shape_for(self, input_shapes):
        rnn_shape, indices_shape = input_shapes
        return super(StackLSTM, self).get_output_shape_for(rnn_shape)

    def compute_mask(self, input, mask):
        if self.return_sequences:
            if isinstance(mask, list):
                return mask[0]
            return mask
        else:
            return None
    
    def call(self, xpind, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        x, indices = xpind
        if isinstance(mask, list):
            mask, _ = mask
        input_shape = self.input_spec[0].shape
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
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.stack_rnn(self.step, 
                                                   preprocessed_input,
                                                   initial_states, 
                                                   indices,
                                                   go_backwards=self.go_backwards,
                                                   mask=mask,
                                                   constants=constants,
                                                   unroll=self.unroll,
                                                   input_length=input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))
            self.cached_states = states

        if self.return_sequences:
            return outputs
        else:
            return last_output



class FancyDense(Dense):
    def __init__(self, fancy_W, middle_dim=None, out_dim=None, *args, **kwargs):
        self.fancy_W = fancy_W
        if middle_dim is None:
            assert hasattr(self.fancy_W, '_keras_shape'), "In FancyDense, the fancyW does not have a keras shape"
            assert self.fancy_W.ndim == 2, "In FancyDense, the fancyW has the wrong number of dims"
            output_dim, middle_dim = self.fancy_W._keras_shape

        super(FancyDense, self).__init__(output_dim, *args, **kwargs)

        self.middle_dim = middle_dim
        #self.output_dim = output_dim

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        # came in as an embedding.. leaving as an output
        self.W = self.init((input_dim, self.middle_dim),
                           name='{}_W'.format(self.name))
        if self.bias:
            self.b = K.zeros((self.output_dim,),
                             name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.fancy_W, self.b]
        else:
            self.trainable_weights = [self.W, self.fancy_W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer and self.bias:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.b_constraint and self.bias:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        out_W = K.permute_dimensions(self.fancy_W, (1,0))
        output = K.dot(K.dot(x, self.W), out_W)
        if self.bias:
            output += self.b
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'input_dim': self.input_dim,
                  'middle_dim': self.middle_dim,
                  'bias': self.bias}
        base_config = super(FancyDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DataLayer(Layer):
    '''TODO: dosctring
    '''
    def __init__(self, input_tensor, input_dtype=None, name=None):
        self.input_spec = None
        self.supports_masking = False
        self.uses_learning_phase = False
        self.trainable = False
        self.built = True
        self.ignore_me = True

        self.inbound_nodes = []
        self.outbound_nodes = []

        self.trainable_weights = []
        self.non_trainable_weights = []
        self.regularizers = []
        self.constraints = {}

        if not name:
            prefix = 'static'
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name

        if not input_dtype:
            input_dtype = K.floatx()

        self.tensor = K.variable(input_tensor, dtype=input_dtype, name=name)
        self.batch_input_shape = input_tensor.shape
        self.tensor._keras_shape = input_tensor.shape
        self.tensor._keras_history = (self, 0, 0)
        self.tensor._uses_learning_phase = False

        Node(self,
             inbound_layers=[],
             node_indices=[],
             tensor_indices=[],
             input_tensors=[self.tensor],
             output_tensors=[self.tensor],
             input_masks=[None],
             output_masks=[None],
             input_shapes=[self.batch_input_shape],
             output_shapes=[self.batch_input_shape])

    def get_config(self):
        config = {'batch_input_shape': self.batch_input_shape,
                  'input_dtype': self.input_dtype,
                  'name': self.name}
        return config

