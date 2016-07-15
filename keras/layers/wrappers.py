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

            last_output, outputs, states = K.rnn(step, X,
                                                 initial_states=[])
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
