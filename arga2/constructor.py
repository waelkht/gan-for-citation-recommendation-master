from model import ARGA, ARVGA, Discriminator
from optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data
import numpy as np
import tensorflow as tf
from preprocessing import make_train_eval_test, construct_feed_dict

flags = tf.app.flags
FLAGS = flags.FLAGS

def get_placeholder(num_nodes):
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'labels': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'real_distribution': tf.placeholder(dtype=tf.float32, shape=[num_nodes, FLAGS.hidden2],
                                            name='real_distribution')

    }

    return placeholders

"""
Creates a Model (ARGA_VAE or ARGA_AE)
#### INPUT ####
:model_str         = eg. 'arga_vae'
:placeholders:     A Dictionary of tf.placeholders: ['features'[Nxm],'labels'[Nxn], dropout'(float), 'real_distribution'[Nx32]]
:num_features:     The number of features = m
:num_nodes:        The number of Nodes N
:features_nonzero: The number of entries in the feature matrix which are not zero
#### OUTPUT ####
:d_real            An output value wich indicates if the input was real or not
:discriminator     the dicriminator object (not linked to the AE)
:model             The Model (ARGA_VAE or ARGA_AE)
"""
def get_model(model_str, placeholders, num_features, num_nodes, features_nonzero):
    
    # get a Discriminator.
    discriminator = Discriminator()
    # construct the discriminator with the real distribution [N,32] as input and one value as output
    d_real = discriminator.construct(placeholders['real_distribution'])

    if model_str == 'arga_ae':
        model = ARGA(placeholders, num_features, features_nonzero)
    elif model_str == 'arga_vae':
        model = ARVGA(placeholders, num_features, num_nodes, features_nonzero)
    else:
        raise ValueError('Unknown model: {}'.format(model))
    

    return d_real, discriminator, model


"""
##### INPUT #####
:data_name = 'cora'
##### OUTPUT #####
# An dictionary with the following fields:
:labels           The labels [N x n]: csr format
:num_features     The number of features: m
:num_nodes        The number of nodes: N
:pos_weight       The position weight of the labels
:norm             The norm of the labels
:features         # features = (coord, values, shape) of features
:train            The training labels: [N x ?]
:val              The evaluation labels: [N x ?]
:val_false        The false evalutation labels: [N x ?]
:test             The test labels: [N x ?]
:test_false       The false test labels: [N x ?]
:features_nonzero The number of non zero entries in the feature matrix
:divisions        The feature matrix may contain multiple sources (authors, venues, ...). divisions is an array
                  of integers indicating the width of each source in the matrix.
"""
def format_data(data_name):
    # Load data
    
    # labels:       The labeled Data [N x n]
    # features:     Feature Matrix [ N x m ]. Type: scipy.sparse.csr.csr_matrix
    # tetd:         A dict: <row-in-adj-mat> ==> 'test' / 'train' / 'eval'
    labels, features, tetd, divisions = load_data(data_name)

    train, val, val_false, test, test_false, train_mask, val_mask, test_mask = make_train_eval_test(labels, tetd)

    num_nodes = labels.shape[0] # num_nodes = N

    num_features = features.shape[1]
    labels_nonzero = len(labels.tocoo().data) # the number of non_zero values in the label matrix
    features_nonzero = len(features.tocoo().data) # the number of non_zero values in the featue matrix

    # The position weight is the ration between positive and negative Examples:
    # E.g. We have a total of 4690 instances where Class P hase 123 members, Class Q has 4567 members then
    # the weight for the positive examples (P) has to be 4567/123 = 37.13
    pos_weight = float(labels.shape[0] * labels.shape[1] - labels_nonzero) / labels_nonzero # pos_weight = (N*N-sum(A)) / sum(A)
    
#     col_weight = (np.sum(train) / np.sum(train, axis = 0)).astype(np.float32)
#     col_weight = np.clip(col_weight, 0, train.shape[1])
#     col_weight = col_weight / np.sum(col_weight)
    
    col_weight = np.sum(train, axis=0)
    col_weight = 1 / (col_weight+1e-7)
     
    items = [labels, num_features, num_nodes, pos_weight, col_weight, features, train, val, val_false, test, test_false, features_nonzero, train_mask, val_mask, test_mask, divisions]
    names = ['labels', 'num_features', 'num_nodes', 'pos_weight', 'col_weight', 'features', 'train', 'val', 'val_false', 'test', 'test_false', 'features_nonzero', 'train_mask', 'val_mask', 'test_mask', 'divisions']
    
    feas = {}
    for i, item in enumerate(items):
        feas[names[i]] = item
        
    return feas
"""
Create an optimizer for the model (ARGA_VAE or ARGA_AE)
#### INPUTS ####
:model_str         might be 'arga_ae' or 'arga_vae'
:model             The model itself as an object
:discriminator     The discriminator as an object
:placeholders      The placeholders ['features'[Nxm],'adj'[NxN], 'adj_orig'[NxN], 'dropout'(float), 'real_distribution'[Nx32]]
:pos_weight        The pos_weight of adj
:norm              The norm of adj
:d_real            The output float of the discriminator indicating if the input is real or not
:num_nodes         The number of nodes: N
:num_features      The number of features: m
:train_mask        The train maks, indicating the nodes which belong to the training data
#### OUTPUTS ####
:opt:              The optimizer object
"""
def get_optimizer(model_str, model, discriminator, placeholders, pos_weight, col_weight, d_real, num_nodes, num_features, train_mask):
    
    if model_str == 'arga_ae':
        # build a second discriminator with the same node,reconstructions
        # but as input we use the embeddings of the model!
        d_fake = discriminator.construct(model.embeddings, reuse=True)
        opt = OptimizerAE(preds = model.reconstructions,
                          labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['labels'],
                                                                        validate_indices=False), [-1, num_features]),
                          pos_weight = pos_weight,
                          col_weight = col_weight,
                          d_real = d_real,
                          d_fake = d_fake,
                          train_mask = train_mask)
        
    elif model_str == 'arga_vae':
        d_fake = discriminator.construct(model.embeddings, reuse=True)
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels = tf.sparse_tensor_to_dense(placeholders['labels'],
                                                              validate_indices=False),
                           model = model,
                           num_nodes = num_nodes,
                           pos_weight = pos_weight,
                           d_real = d_real,
                           d_fake = d_fake)
    else:
        raise ValueError("Unknown model: {}".model_str)
    
    return opt

"""
Runs one Step of the network.
The (V)AE is optimized 5 times.
The Generator and the Discriminator only once.
#### INPUTS ####
:model            The model object; e.g ae_model
:opt              The optimizer for model. It holds  opt.generator_optimizer opt.optimizer and opt.grads_vars
:sess             The tf.Session()
:labels           The labels
:features         sparse represenation of the feature matrix X_0 = (coord, values, shape)
:placeholders     placeholders: {'features'[Nxm],'labels'[NxN], 'dropout'(float), 'real_distribution'[Nx32]}
##### OUTPUT #####
:re:              The reconstructions of the (V)AE [Nx32]
:avg_cost:        The cost of the (V)AE
:d_loss           The loss of the discriminator
:g_loss           The loss of the generator
"""
def update(model, opt, sess, labels, features, placeholders):
    
    # Construct feed dictionary:
    feed_dict = construct_feed_dict(features, labels, FLAGS.dropout, placeholders)

    emb = sess.run(model.embeddings, feed_dict=feed_dict) # emb = [N x 32]
    rec = sess.run(model.reconstructions, feed_dict=feed_dict)
    # _labels = label.tocoo()
    # __labels = (zip(_label.row, _label.col), _label.data, _label.shape)
    # _features = features.tocoo()
    # __features = (zip(_feature.row, _feature.col), _feature.data, _feature.shape)

    # Get a normal distribution on [N x 32] with \my=0 and \sigma=1
    z_real_dist = np.random.randn(labels.shape[0], FLAGS.hidden2)
    
    # Set the placeholder for the real_distribution
    feed_dict.update({placeholders['real_distribution']: z_real_dist})

    # self.opt_op         is the reconstruction loss (based on opt.cost)
    # opt.dc_loss         is the discriminator loss (based on opt.discriminator_optimizer)
    # opt.generator_loss  is the generator_loss (based on the opt.generator_optimizer)
    for _ in range(5):
        _, reconstruct_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
    d_loss = 0.
    g_loss = 0.
    d_loss, _ = sess.run([opt.dc_loss, opt.discriminator_optimizer], feed_dict=feed_dict)
    g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer], feed_dict=feed_dict)

    avg_cost = reconstruct_loss

    # return the embeddings and the cost of the vae.
    return rec, avg_cost, d_loss, g_loss
    # [9, 10, 14, 16, 17, 52, 53, 57, 71, 73, 74, 75, 76, 77, 
    # [rec[9, x] for x in labels[9].tocoo().col]
