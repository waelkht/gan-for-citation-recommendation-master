from initializations import *
import tensorflow as tf
from tensorflow.python.estimator import inputs
from tensorflow.python.ops.variable_scope import AUTO_REUSE

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    :x                    the input values which have to be dopped randomly
    :keep_prob            The probability that a link will not be droped
    :num_nonzero_elems    The number of entries in x
    :returns:             x, but some values are dropped. x is also normalized again.
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    # drophout_mask = (true/false) * num_nonzero_elems
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    # apply the mask on the input x
    pre_out = tf.sparse_retain(x, dropout_mask)
    # normalize the output again via (1./keep_prob)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights") # [m, 32]
        self.dropout = dropout
        self.act = act

    """
    Calls the layer
    :inputs        = (coord, values, shape) of the *sparse* feature matrix. It has the shape [N x N]
    :return:       the output = p(weights[N x m]*drop * inputs[m x 32])
    """
    def _call(self, inputs): 
        x = inputs
        x = tf.nn.dropout(x, 1.-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        outputs = self.act(x)
        return outputs

class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero
        
    """
    Calls the layer
    :inputs The features [N, m]
    :return: the output = p([N x N] * [N x m]*drop * [m x 32])
    """
    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1.-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        outputs = self.act(x)
        return outputs
    
class Dense(Layer):
    """
    A normal dense layer with droput
    """
    def __init__(self, input_dim, output_dim, dropout, act, **kwargs):
        super(Dense, self).__init__(**kwargs)
        name = kwargs.get('name')
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._dropout = dropout
        self._act = act
        self.vars['weights'] = tf.get_variable("weights{}".format(name), shape=[input_dim, output_dim], initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        self.vars['bias'] = tf.get_variable("bias{}".format(name), shape=[output_dim], initializer=tf.constant_initializer(0.0))
        
    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1.-self._dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.add(x, self.vars['bias'])
        outputs = self._act(x)
        return outputs
