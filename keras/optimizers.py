from __future__ import absolute_import
from . import backend as K
import numpy as np
from .utils.generic_utils import get_from_module
from six.moves import zip


def clip_norm(g, c, n):
    if c > 0:
        g = K.switch(n >= c, g * c / n, g)
    return g


def kl_divergence(p, p_hat):
    return p_hat - p + p * K.log(p / p_hat)


class Optimizer(object):
    '''Abstract optimizer base class.

    Note: this is the parent class of all optimizers, not an actual optimizer
    that can be used for training models.

    All Keras optimizers support the following keyword arguments:

        clipnorm: float >= 0. Gradients will be clipped
            when their L2 norm exceeds this value.
        clipvalue: float >= 0. Gradients will be clipped
            when their absolute value exceeds this value.
    '''
    def __init__(self, **kwargs):
        allowed_kwargs = {'clipnorm', 'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise Exception('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []

    def get_state(self):
        return [K.get_value(u[0]) for u in self.updates]

    def set_state(self, value_list):
        assert len(self.updates) == len(value_list)
        for u, v in zip(self.updates, value_list):
            K.set_value(u[0], v)

    def get_updates(self, params, constraints, loss):
        raise NotImplementedError

    def get_gradients(self, loss, params):
        grads = K.gradients(loss, params)
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads

    def set_weights(self, weights):
        '''Sets the weights of the optimizer, from Numpy arrays.

        Should only be called after computing the gradients
        (otherwise the optimizer has no weights).

        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the optimizer (i.e. it should match the
                output of `get_weights`).
        '''
        params = self.weights
        if len(params) != len(weights):
            raise Exception('Provided weight array does not match  weights (' +
                            str(len(params)) + ' optimizer params vs. ' +
                            str(len(weights)) + ' provided weights)')
        for p, w in zip(params, weights):
            if K.get_value(p).shape != w.shape:
                raise Exception('Optimizer weight shape ' +
                                str(K.get_value(p).shape) +
                                ' not compatible with '
                                'provided weight shape ' + str(w.shape))
            K.set_value(p, w)

    def get_weights(self):
        '''Returns the current weights of the optimizer,
        as a list of numpy arrays.
        '''
        weights = []
        for p in self.weights:
            weights.append(K.get_value(p))
        return weights

    def get_config(self):
        config = {'name': self.__class__.__name__}
        if hasattr(self, 'clipnorm'):
            config['clipnorm'] = self.clipnorm
        if hasattr(self, 'clipvalue'):
            config['clipvalue'] = self.clipvalue
        return config



class SGD(Optimizer):
    '''Stochastic gradient descent, with support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    '''
    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0.)
        self.lr = K.variable(lr)
        self.momentum = K.variable(momentum)
        self.decay = K.variable(decay)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        lr = self.lr * (1. / (1. + self.decay * self.iterations))
        self.updates = [(self.iterations, self.iterations + 1.)]

        # momentum
        self.weights = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        for p, g, m in zip(params, grads, self.weights):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append((m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append((p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SARSA_SGD(SGD):
    '''SARSA update technique for reinforcement learning applications. 

    # Arguments
    '''
    def __init__(self, trace_decay=0.9, time_decay=0.9, *args, **kwargs):
        super(SARSA_SGD, self).__init__(*args, **kwargs)
        self.trace_decay = trace_decay
        self.time_decay = time_decay
        self.e_trace = None

    def get_gradients(self, loss, params):
        grads = super(SARSA_SGD, self).get_gradients(loss, params)
        out_grads = []
        for p,g in zip(params, grads):
            if p.name == "value_approximation_W":
                # lambda * gamma * e
                self.e_trace = K.variable(np.zeros(K.get_value(p).shape))
                self.e_update = self.trace_decay * self.time_decay * self.e_trace
                # + dV/dw or dQ/dw ; divide by td to recover it from MSE
                self.e_update += g/self.td
                g = self.td*self.e_update
            out_grads.append(g)
        return out_grads

    def set_td(self, td_tensor):
        """ gives the optimizer a handle on the Temporal Difference error

        # Arguments
            td_tensor: keras tensor.  Output from TD calculations. 
        """
        self.td = td_tensor

    def get_updates(self, params, constraints, loss):
        updates = super(SARSA_SGD, self).get_updates(params, constraints, loss)
        if self.e_trace is not None:
            updates.append((self.e_trace, self.e_update))

class RMSprop(Optimizer):
    '''RMSProp optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).

    This optimizer is usually a good choice for recurrent
    neural networks.

    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.
    '''
    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, **kwargs):
        super(RMSprop, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = K.variable(lr)
        self.rho = K.variable(rho)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        # accumulators
        self.weights = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        self.updates = []

        for p, g, a in zip(params, grads, self.weights):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append((a, new_a))
            new_p = p - self.lr * g / K.sqrt(new_a + self.epsilon)

            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append((p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': float(K.get_value(self.rho)),
                  'epsilon': self.epsilon}
        base_config = super(RMSprop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SARSA_RMSprop(Optimizer):
    '''SARSA update technique for reinforcement learning applications. 

    # Arguments
    '''
    def __init__(self, trace_decay=0.9, time_decay=0.9, *args, **kwargs):
        super(SARSA_RMSprop, self).__init__(*args, **kwargs)
        self.e_trace = None
        self.trace_decay = trace_decay
        self.time_decay = time_decay

    def get_gradients(self, loss, params):
        grads = super(SARSA_RMSprop, self).get_gradients(loss, params)
        out_grads = []
        for p,g in zip(params, grads):
            if p.name == "value_approximation_W":
                # lambda * gamma * e
                self.e_trace = K.variable(np.zeros(K.get_value(p).shape))
                self.e_update = self.trace_decay * self.time_decay * self.e_trace
                # + dV/dw or dQ/dw ; divide by td to recover it from MSE
                self.e_update += g/self.td
                g = self.td*self.e_update
            out_grads.append(g)
        return out_grads

    def set_td(self, td_tensor):
        """ gives the optimizer a handle on the Temporal Difference error

        # Arguments
            td_tensor: keras tensor.  Output from TD calculations. 
        """
        self.td = td_tensor

    def get_updates(self, params, constraints, loss):
        updates = super(SARSA_RMSprop, self).get_updates(params, constraints, loss)
        if self.e_trace is not None:
            updates.append((self.e_trace, self.e_update))

class Adagrad(Optimizer):
    '''Adagrad optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0.
    '''
    def __init__(self, lr=0.01, epsilon=1e-6, **kwargs):
        super(Adagrad, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = K.variable(lr)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        # accumulators
        self.weights = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        self.updates = []

        for p, g, a in zip(params, grads, self.weights):
            new_a = a + K.square(g)  # update accumulator
            self.updates.append((a, new_a))
            new_p = p - self.lr * g / K.sqrt(new_a + self.epsilon)
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append((p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'epsilon': self.epsilon}
        base_config = super(Adagrad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Adadelta(Optimizer):
    '''Adadelta optimizer.

    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate.
            It is recommended to leave it at the default value.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701)
    '''
    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, **kwargs):
        super(Adadelta, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = K.variable(lr)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        accumulators = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        delta_accumulators = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        self.weights = accumulators + delta_accumulators
        self.updates = []

        for p, g, a, d_a in zip(params, grads, accumulators, delta_accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append((a, new_a))

            # use the new accumulator and the *old* delta_accumulator
            update = g * K.sqrt(d_a + self.epsilon) / K.sqrt(new_a + self.epsilon)

            new_p = p - self.lr * update
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append((p, new_p))

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * K.square(update)
            self.updates.append((d_a, new_d_a))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': self.rho,
                  'epsilon': self.epsilon}
        base_config = super(Adadelta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Adam(Optimizer):
    '''Adam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, **kwargs):
        super(Adam, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations + 1)]

        t = self.iterations + 1
        lr_t = self.lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

        ms = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        vs = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        self.weights = ms + vs
        import theano
        debugprints = []
        F_db = theano.printing.Print("Debugprint")

        for p, g, m, v in zip(params, grads, ms, vs):
            debugprints.append(F_db(p))
            debugprints.append(F_db(g))
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append((m, m_t))
            self.updates.append((v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append((p, new_p))
        return self.updates #, debugprints

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon}
        base_config = super(Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SARSA_Adam(Adam):
    '''SARSA update technique for reinforcement learning applications. 

    # Arguments
    '''
    def __init__(self, trace_decay=0.9, time_decay=0.9, *args, **kwargs):
        super(SARSA_Adam, self).__init__(*args, **kwargs)
        self.e_trace = None
        self.trace_decay = trace_decay
        self.time_decay = time_decay

    def get_gradients(self, loss, params):
        grads = super(SARSA_Adam, self).get_gradients(loss, params)
        out_grads = []
        for p,g in zip(params, grads):
            if p.name == "value_approximation_W":
                # lambda * gamma * e
                self.e_trace = K.variable(np.zeros(K.get_value(p).shape))
                self.e_update = self.trace_decay * self.time_decay * self.e_trace
                # + dV/dw or dQ/dw ; divide by td to recover it from MSE
                self.e_update += g/self.td
                g = self.td*self.e_update
            out_grads.append(g)
        return out_grads

    def set_td(self, td_tensor):
        """ gives the optimizer a handle on the Temporal Difference error

        # Arguments
            td_tensor: keras tensor.  Output from TD calculations. 
        """
        self.td = td_tensor

    def get_updates(self, params, constraints, loss):
        updates = super(SARSA_Adam, self).get_updates(params, constraints, loss)
        if self.e_trace is not None:
            updates.append((self.e_trace, self.e_update))

class WoLF(SARSA_Adam):
    '''Gradient Win or Lose Fast (GraWoLF) algorithm by Bowling and Veloso
       This implementation wraps the SARSA_Adam, which means it will use the Adam
       optimizer for updating parameters, and the SARSA algorithm for managing
       an eligibility trace for the function approximation of the critic. 

    # Arguments
        beta: 0 < float < 1.  
              the averaging rate of the policy parameter vector
              a higher value means the running average will use a shorter history

    # References:
        [Simultaneous Adversarial Multi-Robot Learning]()
    '''
    def __init__(self, avg_rate=0.1, *args, **kwargs):
        self.avg_rate = avg_rate
        super(WoLF, self).__init__(*args, **kwargs)

    def set_pi(self, pi):
        self.pi = pi
        
    def get_updates(self, params, constraints, loss):
        updates = super(WoLF, self).get_updates(params, constraints, loss)
        out_updates = []
        for param, update in updates:
            if param.name == "average_pi_W":
                theta = self.pi.W
                assert theta.name == "pi_W"
                update = (1.0-self.avg_rate)*param+self.avg_rate*self.theta
            out_updates.append((param, update))
        return out_updates
    

class Adamax(Optimizer):
    '''Adamax optimizer from Adam paper's Section 7. It is a variant
     of Adam based on the infinity norm.

    Default parameters follow those provided in the paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, **kwargs):
        super(Adamax, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0.)
        self.lr = K.variable(lr)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [(self.iterations, self.iterations + 1)]

        t = self.iterations + 1
        lr_t = self.lr / (1. - K.pow(self.beta_1, t))

        # zero init of 1st moment
        ms = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        # zero init of exponentially weighted infinity norm
        us = [K.variable(np.zeros(K.get_value(p).shape)) for p in params]
        self.weights = ms + us

        for p, g, m, u in zip(params, grads, ms, us):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            u_t = K.maximum(self.beta_2 * u, K.abs(g))
            p_t = p - lr_t * m_t / (u_t + self.epsilon)

            self.updates.append((m, m_t))
            self.updates.append((u, u_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append((p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon}
        base_config = super(Adamax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# aliases
sgd = SGD
rmsprop = RMSprop
adagrad = Adagrad
adadelta = Adadelta
adam = Adam
adamax = Adamax


def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'optimizer',
                           instantiate=True, kwargs=kwargs)
