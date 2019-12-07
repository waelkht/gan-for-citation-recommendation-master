from layers import GraphConvolution, GraphConvolutionSparse, Dense
import tensorflow as tf
from tensorflow.python.debug.examples.debug_tflearn_iris import _IRIS_INPUT_DIM

flags = tf.app.flags
FLAGS = flags.FLAGS


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
    :placeholders      A Dictionary of tf.placeholders: ['features'[Nxm], 'lables'[Nxn], 'dropout'(float), 'real_distribution'[N,32]]
    :num_features      The number of features = m
    :features_nonzero  The number of entries in the feature-matrix which are non-zero
    """
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(ARGA, self).__init__(**kwargs)

        self.inputs = placeholders['features'] # sparse representation of the feature matrix X_0 = (coord, values, shape)
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.labels = placeholders['labels']  # sparse representation of the feature labels Y = (coord, values, shape)
        self.dropout = placeholders['dropout']
        self.build()

    """
    This function is called by the super-constructor. It builds the model.
    """
    def _build(self):

        with tf.variable_scope('Encoder', reuse=None):
            # define a Sparse Convolution [Nxm] => [Nx32]
            # The input (X^0) is dropped, multiplied by A^0 and W^0 and then activated via. relu
            self._hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden1,
                                                  features_nonzero=self.features_nonzero,
                                                  act=lambda x: x,
                                                  dropout=self.dropout,
                                                  logging=self.logging,
                                                  name='e_dense_1')(self.inputs)

            # define a sparse convolution [Nx32] => [Nx32]
            # The input (X^1) is dropped, multiplied by A^1 and W^1 and then activated via. relu
            self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging,
                                           name='e_dense_2')(self._hidden1)
                                            
            # self.embeddings = tf.nn.softmax(self.embeddings, axis = 1)
 
            self._hidden2 = Dense(input_dim = FLAGS.hidden2,
                                  output_dim= FLAGS.hidden1,
                                  act = lambda x: x,
                                  dropout= self.dropout,
                                  logging = self.logging,
                                  name = "e_dense_3")(self.embeddings)                            
            
            self.reconstructions = Dense(input_dim = FLAGS.hidden1,
                                         output_dim= self.input_dim,
                                         act = lambda x: x,
                                         dropout= self.dropout,
                                         logging = self.logging,
                                         name = "e_dense_4")(self._hidden2)


class ARVGA(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(ARVGA, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        
        with tf.variable_scope('Encoder'):
            # Graph Convolution:
            # [NxN] * [Nxm] * [mx32] => [Nx32]
            self._hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                  output_dim=FLAGS.hidden1,
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging,
                                                  name='e_dense_1')(self.inputs)

            # Get the means
            # [NxN] * [Nx32] * [mx32] => [Nx32]
            self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging,
                                           name='e_dense_2')(self._hidden1)

            # Get the std_dev
            # [NxN] * [Nx32] * [mx32] => [Nx32]
            self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                              output_dim=FLAGS.hidden2,
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              logging=self.logging,
                                              name='e_dense_3')(self._hidden1)

            # calculate the variation of the embeddings:
            # Embedd = mean + rand_norm * std_dev
            self.embeddings = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)

            # Create the reconstructions
            self._hidden2 = Dense(input_dim = FLAGS.hidden2,
                                  output_dim= FLAGS.hidden1,
                                  act = tf.nn.relu,
                                  dropout= self.dropout,
                                  logging=self.logging,
                                  name = "e_dense_3")(self.embeddings)
            
            self.reconstructions = Dense(input_dim = FLAGS.hidden1,
                                         output_dim= self.input_dim,
                                         act = tf.nn.relu,
                                         dropout= self.dropout,
                                         logging=self.logging,
                                         name = "e_dense_3")(self._hidden2)


# Define a discriminator 
class Discriminator(Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

    """
    Construct a Discriminator with the buildup: 32-64-16-1
    :inputs  A tf.placeholder of size [N,32]
    :return: The output layer which is only one float (true/false)
    """
    def construct(self, inputs, reuse = False):
        # with tf.name_scope('Discriminator'):
        with tf.variable_scope('Discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            tf.set_random_seed(1)
            dc_den1 = Dense(input_dim= FLAGS.hidden2,
                            output_dim = FLAGS.hidden3,
                            dropout = FLAGS.dropout,
                            act = lambda x: x,
                            logging = self.logging,
                            name = 'dc_den1')(inputs)
                            
            dc_den2 = Dense(input_dim= FLAGS.hidden3,
                            output_dim = FLAGS.hidden1,
                            dropout = FLAGS.dropout,
                            act = lambda x: x,
                            logging = self.logging,
                            name = 'dc_den2')(dc_den1)
                            
            output = Dense(input_dim= FLAGS.hidden1,
                            output_dim = 1,
                            dropout = FLAGS.dropout,
                            act = tf.nn.sigmoid,
                            logging = self.logging,
                            name = 'dc_den3')(dc_den2)
                    
            #dc_den1 = tf.nn.relu(dense(inputs, FLAGS.hidden2, FLAGS.hidden3, name='dc_den1'))
            # dc_den2 = tf.nn.relu(dense(dc_den1, FLAGS.hidden3, FLAGS.hidden1, name='dc_den2'))
            #output = dense(dc_den2, FLAGS.hidden1, 1, name='dc_output')
            
            return output   
