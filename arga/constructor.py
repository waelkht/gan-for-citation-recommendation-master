from model import ARGA, ARVGA, Discriminator, Generator
from optimizer import OptimizerAE, OptimizerVAE
import scipy.sparse as sp
from input_data import load_data
import inspect
import numpy as np
import tensorflow as tf
from preprocessing import preprocess_graph, sparse_to_tuple, my_make_test_edges, construct_feed_dict

flags = tf.app.flags
FLAGS = flags.FLAGS

def get_placeholder(adj):
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'real_distribution': tf.placeholder(dtype=tf.float32, shape=[adj.shape[0], FLAGS.hidden2],
                                            name='real_distribution')

    }

    return placeholders

"""
Creates a Model (ARGA_VAE or ARGA_AE)
#### INPUT ####
:model_str         = eg. 'arga_vae'
:placeholders:     A Dictionary of tf.placeholders: ['features'(float),'adj'(float), 'adj_orig'(float), 'dropout'(float), 'real_distribution'[N,32]]
:num_features:     The number of features = m
:num_nodes:        The number of Nodes N
:features_nonzero: The number of entries in the feature matrix which are not zero
#### OUTPUT ####
:d_real            An output value wich indicates if the input was real or not
:discriminator     the dicriminator object (not linked to the AE)
:model             The Model (ARGA_VAE or ARGA_AE)
"""
def get_model(model_str, placeholders, num_features, num_nodes, features_nonzero):
    
    # get a Generator
    generator = Generator()
    fake_emb = generator.construct(placeholders['real_distribution'])
    
    # get a Discriminator.
    discriminator = Discriminator()
    # construct the discriminator with the real distribution [N,32] as input and one value as output
    d_real = discriminator.construct(fake_emb)
    
    model = None
    if model_str == 'arga_ae':
        model = ARGA(placeholders, num_features, features_nonzero)

    elif model_str == 'arga_vae':
        model = ARVGA(placeholders, num_features, num_nodes, features_nonzero)

    return d_real, discriminator, generator, model


"""
##### INPUT #####
:data_name = 'cora'
##### OUTPUT #####
# An dictionary with the following fields:
:adj:             The adjacency matrix, but withouth the eval and the test edges
:num_features:    The number of features = m
:num_nodes:       The number of Nodes N
:features_nonzero The number of entries in the feature matrix which ar not zero
:pos_weight:      The position weight od adj
:norm:            The norm of adj
:adj_norm:        The preprocessed adj: = D^-0.5 * (adj+I) * D^-0.5
:adj_label:       sparse represenation of (adj + I) = (coord, values, shape)
:features:        sparse represenation of the feature matrix X_0 = (coord, values, shape)
:true_labels:     A mask (array of bools) where y labels are available
:train_edges:     All edges in adj_train, but only uni-directional
:val_edges:       10% of the edges in adj But only uni-directional
:val_edges_false: as many edges as in val_edges, but the are fake (not contained in adj !)
:test_edges       5% of the edges in adj. Bun only uni-directional
:test_edges_false test_edges_false: as many edges as in test_edges, but the are fake (not contained in adj !)
:adj_orig:        The original adj. matrix (no edges dropped) but the diagonals are zero
"""
def format_data(data_name):
    # Load data
    
    # adj:        Adjacency Matrix [ N x N ]. Type: scipy.sparse.csr.csr_matrix
    # features:   Feature Matrix [ N x m ]. Type: scipy.sparse.csr.csr_matrix
    # tetd:        A dict: <row-in-adj-mat> ==> 'test' / 'train' / 'eval'
    adj, features, tetd = load_data(data_name)


    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    # set the diagonal to zero.
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    # since it is a sparse matrix we can remove the zeros to save memory
    adj_orig.eliminate_zeros()

    # adj_train:        like 'adj', but without the edges from eval and test
    # train_edges:      All edges in adj_train
    # val_edges:        some edges from 'adj'. For evaluation purposes
    # val_edges_false:  some fake edges. It has the same size as 'val_edges'
    # test_edges:       some edges from 'adj'. For testing purposes
    # test_edges_false: some fake edges. It has the same size as 'test_edges'
    # mask_train:       has a 1 at pos. n, iff the n-th row of the adj. matrix belongs to train
    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = my_make_test_edges(adj, tetd)
    adj = adj_train

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Add convolution filter to adj_norm
    # A = D^-0.5 * (A+I) * D^-0.5
    adj_norm = preprocess_graph(adj)

    num_nodes = adj.shape[0] # num_nodes = N

    # features = (coord, values, shape) of features
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1] # num_features = m
    features_nonzero = features[1].shape[0] # the number of non_zero values in the feature matrix

    # The position weight is the ration between positive and negative Examples:
    # E.g. We have a total of 3690 instances where Class P hase 123 members, Class Q has 4567 members then
    # the weight for the positive examples (P) has to be 4567/123 = 37.13
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum() # pos_weight = (N*N-sum(A)) / sum(A)
    # the norm is 1/pos_weight
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2) # norm = N*N / (N*N-sum(A))

    adj_label = adj_train + sp.eye(adj_train.shape[0]) # adj_label = adj_train + I
    adj_label = sparse_to_tuple(adj_label) # adj_label = (coord, values, shape) of adj_label
    
    items = [adj, num_features, num_nodes, features_nonzero, pos_weight, norm, adj_norm, adj_label, features, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, adj_orig]
    names = ['adj', 'num_features', 'num_nodes', 'features_nonzero', 'pos_weight', 'norm', 'adj_norm', 'adj_label', 'features', 'train_edges', 'val_edges', 'val_edges_false', 'test_edges', 'test_edges_false', 'adj_orig']
    
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
#### OUTPUTS ####
:opt:              The optimizer object
"""
def get_optimizer(model_str, model, discriminator, placeholders, pos_weight, norm, d_real, num_nodes):
    if model_str == 'arga_ae':
        # build a second discriminator with the same node, but as input we use the embeddings of the model!
        d_fake = discriminator.construct(model.embeddings, reuse=True)
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm,
                          d_real=d_real,
                          d_fake=d_fake)
    elif model_str == 'arga_vae':
        d_fake = discriminator.construct(model.embeddings, reuse=True)
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           d_real=d_real,
                           d_fake=d_fake)
    return opt

"""
Runs one Step of the network.
The (V)AE is optimized 5 times.
The Generator and the Discriminator only once.
#### INPUTS ####
:model            The model object; e.g ae_model
:opt              The optimizer for model. It holds  opt.generator_optimizer opt.optimizer and opt.grads_vars
:sess             The tf.Session()
:adj_norm         The preprocessed adj: D^-0.5 * (adj+I) * D^-0.5
:adj_label        sparse represenation of (adj + I) = (coord, values, shape)
:features         sparse represenation of the feature matrix X_0 = (coord, values, shape)
:placeholders     placeholders: {'features'[Nxm],'adj'[NxN], 'adj_orig'[NxN], 'dropout'(float), 'real_distribution'[Nx32]}
:adj              The adjacency matrix without eval and test edges
##### OUTPUT #####
:emb:             The embeddings of the (V)AE [Nx32]
:avg_cost:        The cost of the (V)AE
"""
def update(model, opt, sess, adj_norm, adj_label, features, placeholders, adj):
    # Construct feed dictionary:
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    feed_dict.update({placeholders['dropout']: 0})    
    # The dictionary has now the form:
    # feed_dict = {placeholders['features']: features,
    #              placeholders['adj']: adj_norm,
    #              placeholders['adj_orig']: adj,
    #              placeholders['dropout']: 0}
    # Now we run the model and receive the embeddings
    emb = sess.run(model.z_mean, feed_dict=feed_dict) # emb = [N x 32]

    # Get a normal distribution on [N x 32] with \my=0 and \sigma=1
    z_real_dist = np.random.normal(loc=0.0, scale=1.0, size=(adj.shape[0], FLAGS.hidden2))
    # z_real_dist = np.random.uniform(low=-1.0, high=1.0, size=(adj.shape[0], FLAGS.hidden2))
    # Set the placeholder for the real_distribution
    feed_dict.update({placeholders['real_distribution']: z_real_dist})

    # Run 5 iterations with the optimizer.
    # opt.opt_op is the optimizer. opt.cost is the value to be minimized
    # In the case of the (V)AE this is the difference of the input X_0 and output X'
    for j in range(5):
        reconstruct_loss, _ = sess.run([opt.encoder_loss, opt.encoder_optimizer], feed_dict=feed_dict)
    # now also optimize the Discriminator ...
    d_loss = 0.
    g_loss = 0.
    # d_loss, _ = sess.run([opt.discriminator_loss, opt.discriminator_optimizer], feed_dict=feed_dict)
    # g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer], feed_dict=feed_dict)

    avg_cost = reconstruct_loss

    # return the embeddings and the cost of the vae.
    return emb, avg_cost, g_loss

# get the name of a variable var in the current frame
def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]
