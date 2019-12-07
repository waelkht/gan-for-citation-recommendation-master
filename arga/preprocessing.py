import numpy as np
import scipy.sparse as sp
import collections

# INPUT:
# sparse_mx: A sparse Matrix 
# Output:
# Coords: An array of coordinates [? x 2] for every non-zero entry in the matrix
# values: The values at the non-zero entries
# shape: A tuple of the form (p, q) representing the shape.
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose() # [? x 2]
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# This function adds a convolution filter to the matrix adj
# A = D^(-0.5) * A * D^(-0.5)
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0]) # A = A + I
    rowsum = np.array(adj_.sum(1)) 
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten()) # get D^(-0.5)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

"""
##### INPUTS #####
:adj_normalized      The preprocessed adj: = D^-0.5 * (adj+I) * D^-0.5
:adj                 sparse represenation of (adj + I) = (coord, values, shape)
:features            sparse represenation of the feature matrix X_0 = (coord, values, shape)
:placeholders        A Dictionary: {'features'[Nxm],'adj'[NxN], 'adj_orig'[NxN], 'dropout'(float), 'real_distribution'[Nx32]}
##### RETURN #####
:feed_dict:          A Dictionary: {placeholders['features']: features, placeholders['adj']:adj_norm, placeholders['adj_orig']: adj}
"""
def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    feed_dict.update({placeholders['features']: features})
    return feed_dict

"""
Function to build test set with X% positive links
NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
##### INPUTS #####
:adj                The ajacency matrix of a graph          
##### RETURN #####
:adj_train:         like adj, but 15% of the edges were randomly removed
:train_edges:       All edges in adj_train, but only uni-directional
:val_edges:         10% of the edges in adj. But only uni-directional
:val_edges_false:   as many edges as in val_edges, but the are fake (not contained in adj !)
:test_edges:        5% of the edges in adj. Bun only uni-directional
:test_edges_false:  as many edges as in test_edges, but the are fake (not contained in adj !)
NOTE: val_edges_false \cup train_edges = \empty AND val_edges_false \cup val_edges = \empty
NOTE: test_edges_false \cup train_edges = \empty AND test_edges_false \cup test_edges = \empty
"""
def make_test_edges(adj):
    
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert adj.diagonal().sum() == 0

    # triu: Return the upper triangular portion of a matrix in sparse format
    adj_triu = sp.triu(adj) # [N x N]
    adj_tuple = sparse_to_tuple(adj_triu) # (coords [p x 2], values [p], shape (N, N))
    edges = adj_tuple[0] # [p x 2]
    edges_all = sparse_to_tuple(adj)[0] # [ (2p) x 2] 
    num_test = int(np.floor(edges.shape[0] / 20.)) # =p/20
    num_val = int(np.floor(edges.shape[0] / 20.)) # =p/20

    all_edge_idx = range(edges.shape[0]) # [p]
    np.random.shuffle(all_edge_idx) # [p]
    val_edge_idx = all_edge_idx[:num_val] # get random indices for the vals
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)] # get another random indices for tests
    test_edges = edges[test_edge_idx] # get test edges
    val_edges = edges[val_edge_idx] # get val edges
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0) # get train edges

    # check if the edge a = [x, y] is contained in the list of edges b = [[p, q], ...]
    def ismember(a, b, tol=5):
#         rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
#         return (np.all(np.any(rows_close, axis=-1), axis=-1) and
#                 np.all(np.any(rows_close, axis=0), axis=0))
#         return a in b
        return a[0] in b[:, 0] and a[1] in b[b[:,0]==a[0],1]


    # this loop adds random edges which are not contained in the adjacency matrix
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0]) # get random i in [0:N]
        idx_j = np.random.randint(0, adj.shape[0]) # get random j in [0:N]
        if idx_i == idx_j:
            continue # ignore edges to itself
        if ismember([idx_i, idx_j], edges_all):
            continue # continue if [i, j] in edges_all
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue # continue if we have it already
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue # continue if we have it already
        test_edges_false.append([idx_i, idx_j]) # otherwise append the negative edge [i, j]
   

    # add random edges to which are not contained in train_edges or val_edges
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

#     assert ~ismember(test_edges_false, edges_all)
#     assert ~ismember(val_edges_false, edges_all)
#     assert ~ismember(val_edges, train_edges)
#     assert ~ismember(test_edges, train_edges)
#     assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix out of training edges
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T # since the adj_train is only 0.85 of the adj_trui we have to transpos it again.

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

"""
Function to build test set with X% positive links
NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
##### INPUTS #####
:adj                The ajacency matrix of a graph       
:ttd                A dictionary: <index-in-adj-matrix> ==> 'test' / 'train' / 'eval'
##### RETURN #####
:adj_train:         like adj, but test and eval edges were removed
:train_edges:       All edges in adj_train, but only uni-directional
:val_edges:         The evaluate edges
:val_edges_false:   as many edges as in val_edges, but the are fake (not contained in adj !)
:test_edges:        The test edges
:test_edges_false:  as many edges as in test_edges, but the are fake (not contained in adj !)
NOTE: It is possible that a false edge may be contained twice in the test_edges_false.
"""
def my_make_test_edges(adj, tetd):
    # remove diagonal
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()    
    
    # (coords [p x 2], values [p], shape (N, N))
    adj_tuple = sparse_to_tuple(adj)
    
    train_edges = []
    val_edges = []
    val_edges_false = []
    test_edges = []
    test_edges_false = []
    
    # divide all the edegs from adj into test and training based on the ttd
    for row, col in adj_tuple[0]:
        if tetd[row] == 'train':
            train_edges.append([row, col])
        elif tetd[row] == 'eval':
            val_edges.append([row, col])
        elif tetd[row] == 'test':
            test_edges.append([row, col])
        else:
            print __file__, " unknown: ", tetd[row]
    
    c = collections.Counter(tetd.values())
    print "Loaded {}(Train)<==>{}(Eval)<==>{}(Test)".format(c['train'], c['eval'], c['test'])
    
    N = adj.shape[0]
    
    # create false test edges. They will be used for performance testing
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, N)
        idx_j = np.random.randint(0, N)
        
        if idx_i == idx_j or adj[idx_i, idx_j] != 0:
            continue 
        else:
            test_edges_false.append([idx_i, idx_j])
    
    # create false eval edges. The will later be used for evaluation       
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, N)
        idx_j = np.random.randint(0, N)
        
        if idx_i == idx_j or adj[idx_i, idx_j] != 0:
            continue
        else:
            val_edges_false.append([idx_i, idx_j])
    
    # format the type from list ==> np.array with size [? x 2]
    train_edges = np.array(train_edges)
    val_edges = np.array(val_edges)
    val_edges_false = np.array(val_edges_false)
    test_edges = np.array(test_edges)
    test_edges_false= np.array(test_edges_false)
    
    # build the new adjacency matrix which does not contain val-edges, nor test-edges!
    train_adj = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), (N, N))
    
    return train_adj, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
