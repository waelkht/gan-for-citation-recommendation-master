from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
_act = tf.nn.tanh

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class ARGA(Model):
    """
    Creates a ARGA Model
    :placeholders      A Dictionary of tf.placeholders: ['features'[Nxm], 'adj'[NxN], 'adj_orig'(float), 'dropout'(float), 'real_distribution'[N,32]]
    :num_features      The number of features = m
    :features_nonzero  The number of entries in the feature-matrix which are non-zero
    """
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(ARGA, self).__init__(**kwargs)

        self.inputs = placeholders['features'] # sparse representation of the feature matrix X_0 = (coord, values, shape)
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj'] # The pre-processed adj: = D^-0.5 * (adj+I) * D^-0.5
        self.dropout = placeholders['dropout']
        self.build()

    """
    This function is called by the super-constructor. It builds the model.
    """
    def _build(self):

        with tf.variable_scope('Encoder', reuse=None):
            # define a Sparse Convolution [Nxm] => [Nx32]
            # The input (X^0) is dropped, multiplied by A^0 and W^0 and then activated via. relu
            
            self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                   output_dim=FLAGS.hidden1,
                                                   adj=self.adj,
                                                   features_nonzero=self.features_nonzero,
                                                   act=_act,
                                                   dropout=self.dropout,
                                                   logging=self.logging,
                                                   name='e_dense_1')(self.inputs)
                                                  
            # add some noise to the hidden layer 1 with standard-deviation=0.1                             
            # self.noise = gaussian_noise_layer(self.hidden1, 0.1)

            # self.softmax = tf.exp(self.hidden1) / tf.reduce_sum(tf.abs(self.hidden1), 1)
            # _sum = tf.reshape(tf.reduce_sum(tf.abs(self.hidden1), 1), [-1, 1]) + (1e-7)
            # self.normalized = tf.multiply(self.hidden1, 1/_sum)
            # self.softmax = tf.nn.softmax(self.hidden1, axis=1)            

            # define a sparse convolution [Nx32] => [Nx32]
            # The input (X^1) is dropped, multiplied by A^1 and W^1 and then activated via. relu
            self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging,
                                           name='e_dense_2')(self.hidden1)

            # since this is no variation encoder we define z_mean as the embeddings
            self.z_mean = self.embeddings

            # Reconstruct the adjacency matrix from the embeddings via inner product
            # [Nx32] => [N^2]
            # NOTE: [N^2] != [N x N]
            self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                        act = lambda x: x,
                                        logging=self.logging)(self.embeddings)




class ARVGA(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(ARVGA, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        with tf.variable_scope('Encoder'):
            # Graph Convolution:
            # [NxN] * [Nxm] * [mx32] => [Nx32]
            self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden1,
                                                  adj=self.adj,
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging,
                                                  name='e_dense_1')(self.inputs)

            # Get the means
            # [NxN] * [Nx32] * [mx32] => [Nx32]
            self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging,
                                           name='e_dense_2')(self.hidden1)

            # Get the std_dev
            # [NxN] * [Nx32] * [mx32] => [Nx32]
            self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                              output_dim=FLAGS.hidden2,
                                              adj=self.adj,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              logging=self.logging,
                                              name='e_dense_3')(self.hidden1)

            # calculate the variation of the embeddings:
            # Embedd = mean + rand_norm * std_dev
            self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)

           
            # Reconstruct the adjacency matrix from the embeddings via inner product
            # [Nx32] => [N^2]
            # NOTE: [N^2] != [N x N]
            self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                          act=lambda x: x,
                                          logging=self.logging)(self.z)
            # set the embeddings
            self.embeddings = self.z


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [n2]
    """
    with tf.variable_scope(name, reuse=None):
        tf.set_random_seed(1)
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out

# Define a discriminator 
class Discriminator(Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.act = _act

    """
    Construct a Discriminator with the buildup: 32-64-32-1
    :inputs  A tf.placeholder of size [N,32]
    :return: The output layer which is only one float (true/false)
    """
    def construct(self, inputs, reuse = False):
        with tf.variable_scope('Discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            tf.set_random_seed(1)
            dc_den1 = self.act(dense(inputs, FLAGS.hidden2, FLAGS.hidden3, name='dc_den1'))
            dc_den2 = self.act(dense(dc_den1, FLAGS.hidden3, FLAGS.hidden1, name='dc_den2'))
            output = dense(dc_den2, FLAGS.hidden1, 1, name='dc_output')
            return output
        
class Generator(Model):
    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.act = _act

    """
    Construct a Discriminator with the buildup: 32-64-32-1
    :inputs  A tf.placeholder of size [N,32]
    :return: The output layer which is only one float (true/false)
    """
    def construct(self, inputs):
        with tf.variable_scope('Generator'):
            tf.set_random_seed(1)
            # gen_den1 = self.act(dense(inputs, FLAGS.hidden2, FLAGS.hidden3, name='gen_den1'))
            # gen_den2 = dense(gen_den1, FLAGS.hidden3, FLAGS.hidden2, name='gen_output')
            # with tf.variable_scope('gen_norm', reuse=None):
            #    std = tf.get_variable("std_dev", shape=[1], initializer=tf.constant_initializer(0.1))
            #    inputs = inputs* (std*std)
            output = inputs
            return output
            
def gaussian_noise_layer(input_layer, std):
    """
    Adds a noise of a normal distribution (mean = 0.0)
    :input_layer the layer for which a random distribution has be build
    :std the standard-deviation
    :return: the input_layer + gaussian noise.
    """
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise       
