from keras import backend as K
from .core import Dense

class SoftAttention(Dense):
    def call(self, x, mask=None):
        score = self.activation(K.dot(x, self.W) + self.b)
        return K.sum(x * K.dimshuffle(score, (0, 'x')), axis=1)
