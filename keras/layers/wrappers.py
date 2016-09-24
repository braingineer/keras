from ..engine import Layer, InputSpec
from .. import backend as K


class Wrapper(Layer):

    def __init__(self, layer, **kwargs):
        self.layer = layer
        self.uses_learning_phase = layer.uses_learning_phase
        super(Wrapper, self).__init__(**kwargs)

    def build(self, input_shape=None):
        '''Assumes that self.layer is already set.
        Should be called at the end of .build() in the
        children classes.
        '''
        self.trainable_weights = getattr(self.layer, 'trainable_weights', [])
        self.non_trainable_weights = getattr(self.layer, 'non_trainable_weights', [])
        self.updates = getattr(self.layer, 'updates', [])
        self.regularizers = getattr(self.layer, 'regularizers', [])
        self.constraints = getattr(self.layer, 'constraints', {})

        # properly attribute the current layer to
        # regularizers that need access to it
        # (e.g. ActivityRegularizer).
        for regularizer in self.regularizers:
            if hasattr(regularizer, 'set_layer'):
                regularizer.set_layer(self)

    def get_weights(self):
        weights = self.layer.get_weights()
        return weights

    def set_weights(self, weights):
        self.layer.set_weights(weights)

    def get_config(self):
        config = {'layer': {'class_name': self.layer.__class__.__name__,
                            'config': self.layer.get_config()}}
        base_config = super(Wrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        from keras.utils.layer_utils import layer_from_config
        layer = layer_from_config(config.pop('layer'))
        return cls(layer, **config)


class TimeDistributed(Wrapper):
    """This wrapper allows to apply a layer to every
    temporal slice of an input.

    The input should be at least 3D,
    and the dimension of index one will be considered to be
    the temporal dimension.

    Consider a batch of 32 samples, where each sample is a sequence of 10
    vectors of 16 dimensions. The batch input shape of the layer is then `(32, 10, 16)`
    (and the `input_shape`, not including the samples dimension, is `(10, 16)`).

    You can then use `TimeDistributed` to apply a `Dense` layer to each of the 10 timesteps, independently:
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
        # now model.output_shape == (None, 10, 8)

        # subsequent layers: no need for input_shape
        model.add(TimeDistributed(Dense(32)))
        # now model.output_shape == (None, 10, 32)
    ```

    The output will then have shape `(32, 10, 8)`.

    Note this is strictly equivalent to using `layers.core.TimeDistributedDense`.
    However what is different about `TimeDistributed`
    is that it can be used with arbitrary layers, not just `Dense`,
    for instance with a `Convolution2D` layer:

    ```python
        model = Sequential()
        model.add(TimeDistributed(Convolution2D(64, 3, 3), input_shape=(10, 3, 299, 299)))
    ```

    # Arguments
        layer: a layer instance.
    """
    def __init__(self, layer, force_reshape=False, **kwargs):
        self.supports_masking = True
        self.force_reshape = force_reshape
        super(TimeDistributed, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        import pdb
        #pdb.set_trace()
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]
        child_input_shape = (input_shape[0],) + input_shape[2:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True
        super(TimeDistributed, self).build()

    def compute_mask(self, x, mask=None):
        if mask is None:
            return None

        input_shape = self.input_spec[0].shape
        m_shape = (input_shape[0]*input_shape[1],) + tuple(K.shape(mask)[2:])
        d_shape = (input_shape[0]*input_shape[1],) + input_shape[2:]
        new_mask = self.layer.compute_mask(K.reshape(x, d_shape), K.reshape(mask, m_shape, ndim=K.ndim(mask)-1))
        if new_mask is None:

            axes = tuple(range(2, K.ndim(mask)))
            return K.any(mask, axes)
            
        out_shape = (input_shape[0], input_shape[1],) + tuple(K.shape(new_mask)[1:])
        out_mask = K.reshape(new_mask, out_shape, ndim=K.ndim(new_mask)+1)
        return out_mask

        '''
        outmask = []
        for i in range(self.input_spec[0].shape[1]):
            mask_i = self.layer.compute_mask(x[:,i], mask[:,i])
            if mask_i is None:
                outmask.append(K.ones_like(mask[:,i]))
            else:
                outmask.append(mask_i)
        outmask = K.pack(tuple(outmask))
        axes = [1, 0] + list(range(2, K.ndim(mask_i)))
        if len(axes) < K.ndim(outmask):
            extra = tuple(range(len(axes), K.ndim(outmask)))
            outmask = K.any(outmask, axis=extra)
            axes = axes[:K.ndim(outmask)]
        try:
            outmask = K.permute_dimensions(outmask, axes)
        except:
            import pdb
            pdb.set_trace()
        return outmask
        '''


    def get_output_shape_for(self, input_shape):
        child_input_shape = (input_shape[0],) + input_shape[2:]
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        timesteps = input_shape[1]
        return (child_output_shape[0], timesteps) + child_output_shape[1:]

    def call(self, X, mask=None):
        input_shape = self.input_spec[0].shape
        if input_shape[0] and not self.force_reshape and mask is None:
            # batch size matters, use rnn-based implementation
            def step(x, states):
                output = self.layer.call(x)
                return output, []
            input_length = input_shape[1]
            if K.backend() == 'tensorflow' and len(input_shape) > 3:
                if input_length is None:
                    raise Exception('When using TensorFlow, you should define '
                                    'explicitly the number of timesteps of '
                                    'your sequences.\n'
                                    'If your first layer is an Embedding, '
                                    'make sure to pass it an "input_length" '
                                    'argument. Otherwise, make sure '
                                    'the first layer has '
                                    'an "input_shape" or "batch_input_shape" '
                                    'argument, including the time axis.')
                unroll = True
            else:
                unroll = False
            last_output, outputs, states = K.rnn(step, X,
                                                 initial_states=[], input_length=input_length, unroll=unroll)
            y = outputs
        else:
            # no batch size specified, therefore the layer will be able
            # to process batches of any size
            # we can go with reshape-based implementation for performance
            input_length = input_shape[1]
            d_shape = (input_shape[0]*input_shape[1],) + input_shape[2:]

            try:
                if mask is not None:
                    mask = K.reshape(safe_mask(mask, X), d_shape)
            except:
                import pdb
                pdb.set_trace()

            X = K.reshape(X, d_shape)  # (nb_samples * timesteps, ...)
            y = self.layer.call(X, mask)  # (nb_samples * timesteps, ...)
            # (nb_samples, timesteps, ...)
            output_shape = self.get_output_shape_for(input_shape)
            y = K.reshape(y, (-1, input_length) + output_shape[2:])
        return y

def safe_mask(mask, x):
    if mask.ndim == x.ndim:
        return K.cast(mask, K.floatx())
    assert mask.ndim == x.ndim - 1
    return K.ones_like(x)*K.cast(K.expand_dims(mask), K.floatx()) ## add 1 broadcastable to end
    raise Exception("This should be impossible???")

Distribute = TimeDistributed

class Bidirectional(Wrapper):
    ''' Bidirectional wrapper for RNNs.

    # Arguments:
        layer: `Recurrent` instance.
        merge_mode: Mode by which outputs of the
            forward and backward RNNs will be combined.
            One of {'sum', 'mul', 'concat', 'ave', None}.
            If None, the outputs will not be combined,
            they will be returned as a list.

    # Examples:

    ```python
        model = Sequential()
        model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
        model.add(Bidirectional(LSTM(10)))
        model.add(Dense(5))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    ```
    '''
    def __init__(self, layer, merge_mode='concat', weights=None, **kwargs):
        if merge_mode not in ['sum', 'mul', 'ave', 'concat', None]:
            raise ValueError('Invalid merge mode. '
                             'Merge mode should be one of '
                             '{"sum", "mul", "ave", "concat", None}')
        self.forward_layer = layer
        config = layer.get_config()
        config['go_backwards'] = not config['go_backwards']
        self.backward_layer = layer.__class__.from_config(config)
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name
        self.merge_mode = merge_mode
        if weights:
            nw = len(weights)
            self.forward_layer.initial_weights = weights[:nw // 2]
            self.backward_layer.initial_weights = weights[nw // 2:]
        self.stateful = layer.stateful
        self.return_sequences = layer.return_sequences
        self.supports_masking = True
        super(Bidirectional, self).__init__(layer, **kwargs)

    def get_weights(self):
        return self.forward_layer.get_weights() + self.backward_layer.get_weights()

    def set_weights(self, weights):
        nw = len(weights)
        self.forward_layer.set_weights(weights[:nw // 2])
        self.backward_layer.set_weights(weights[nw // 2:])

    def get_output_shape_for(self, input_shape):
        if self.merge_mode in ['sum', 'ave', 'mul']:
            return self.forward_layer.get_output_shape_for(input_shape)
        elif self.merge_mode == 'concat':
            shape = list(self.forward_layer.get_output_shape_for(input_shape))
            shape[-1] *= 2
            return tuple(shape)
        elif self.merge_mode is None:
            return [self.forward_layer.get_output_shape_for(input_shape)] * 2

    def call(self, X, mask=None):
        Y = self.forward_layer.call(X, mask)
        Y_rev = self.backward_layer.call(X, mask)
        if self.return_sequences:
            Y_rev = K.reverse(Y_rev, 1)
        if self.merge_mode == 'concat':
            return K.concatenate([Y, Y_rev])
        elif self.merge_mode == 'sum':
            return Y + Y_rev
        elif self.merge_mode == 'ave':
            return (Y + Y_rev) / 2
        elif self.merge_mode == 'mul':
            return Y * Y_rev
        elif self.merge_mode is None:
            return [Y, Y_rev]

    def reset_states(self):
        self.forward_layer.reset_states()
        self.backward_layer.reset_states()

    def build(self, input_shape):
        self.forward_layer.build(input_shape)
        self.backward_layer.build(input_shape)

    def compute_mask(self, input, mask):
        if self.return_sequences:
            if not self.merge_mode:
                return [mask, mask]
            else:
                return mask
        else:
            return None

    @property
    def trainable_weights(self):
        if hasattr(self.forward_layer, 'trainable_weights'):
            return self.forward_layer.trainable_weights + self.backward_layer.trainable_weights
        return []

    @property
    def non_trainable_weights(self):
        if hasattr(self.forward_layer, 'non_trainable_weights'):
            return self.forward_layer.non_trainable_weights + self.backward_layer.non_trainable_weights
        return []

    @property
    def updates(self):
        if hasattr(self.forward_layer, 'updates'):
            return self.forward_layer.updates + self.backward_layer.updates
        return []

    @property
    def regularizers(self):
        if hasattr(self.forward_layer, 'regularizers'):
            return self.forward_layer.regularizers + self.backward_layer.regularizers
        return []

    @property
    def constraints(self):
        _constraints = {}
        if hasattr(self.forward_layer, 'constraints'):
            _constraints.update(self.forward_layer.constraints)
            _constraints.update(self.backward_layer.constraints)
        return _constraints

    def get_config(self):
        config = {"merge_mode": self.merge_mode}
        base_config = super(Bidirectional, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
