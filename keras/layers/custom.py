from keras import backend as K
from .core import Dense, Lambda
from .embeddings import Embedding

class SoftAttention(Dense):
    def call(self, x, mask=None):
        score = self.activation(K.dot(x, self.W) + self.b)
        return K.sum(x * K.dimshuffle(score, (0, 'x')), axis=1)


class BitwiseEmbedding(Lambda):
    """ function for embedding with encoding """
    def __init__(self,  bit_info, *args, **kwargs):
        super(BitwiseEmbed, self).__init__(self.make(bit_info), *args, **kwargs)


    def make(self, bit_info):
        def func(layer_out):
            embeddings = []
            sh = K.shape(layer_out)
            if K.ndim(layer_out) != 2:
                out_shape = sh[:-1] + (0,)
                faceflat = (reduce(mul, out_shape), sh[-1])
                layer_out = K.reshape(layer_out, faceflat)
            else:
                out_shape = (sh[0], 0)

            for i, (in_dim, out_dim) in enumerate(bit_info):
                E = Embedding(input_dim=in_dim, output_dim=out_dim, mask_zero=True)
                embeddings.append(E(layer_out[:,i]))
                out_shape = out_shape[:-1] + (out_shape[-1]+out_dim,)

            merged = merge(embeddings, mode='concat')
            merged = K.reshape(merged, out_shape)

            return merged

        