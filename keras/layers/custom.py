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

        #layer, node_index, tensor_index = self.W._keras_history
        #self.add_inbound_node(layer, node_index, tensor_index)
        
        
    def __call__(self, x, mask=None):
        ### hacky. 
        return super(DynamicEmbedding, self).__call__([x, self.W], mask)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape,_ = input_shape

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
            
        self.built = True

    def compute_mask(self, x, mask=None):
        if isinstance(x, list):
            x,_ = x
        if mask is not None and isinstance(mask, list):
            mask,_ = mask
        return super(DynamicEmbedding, self).compute_mask(x, mask)

    def get_output_shape_for(self, input_shape):
        if isinstance(input_shape, list):
            input_shape,_ = input_shape
        return super(DynamicEmbedding, self).get_output_shape_for(input_shape)

    def call(self, x, mask=None):
        if isinstance(x, list): 
            x,_ = x
        if mask is not None and isinstance(mask, list):
            mask,_ = mask
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
            # tensor abc goes to bac, indexed onto with xyz, goes to xyzac, 
            # x == a, so shape to xayzc == xxyzc
            # take diagonal on first two: xyzc 
            #out = K.colgather()
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







class FancyDense(Dense):
    def __init__(self, fancy_W, middle_dim=None, out_dim=None, *args, **kwargs):
        self.fancy_W = fancy_W
        if middle_dim is None:
            assert hasattr(self.fancy_W, '_keras_shape'), "In FancyDense, the fancyW does not have a keras shape"
            assert self.fancy_W.ndim == 2, "In FancyDense, the fancyW has the wrong number of dims"
            output_dim, middle_dim = self.fancy_W._keras_shape

        super(FancyDense, self).__init__(output_dim, *args, **kwargs)

        self.middle_dim = middle_dim

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
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]

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

    def normalize_mask(self, mask):
        if K.ndim(mask) == 2:
            mask = K.expand_dims(K.all(mask, axis=1))
        elif K.ndim(mask) == 3:
            mask = K.expand_dims(K.all(mask, axis=(-1,-2)))
        return K.cast(mask, K.floatx())

    # imagine x either as (batch x feat) or (batch*time x feat)
    # probably the second one
    # in either case,
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


'''
    

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
'''


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
        self.tensor._sideload = True

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

