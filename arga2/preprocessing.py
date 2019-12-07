import numpy as np
import scipy.sparse as sp
import collections
import random

def dense_to_sparse(matrix):
    """
    This function converts a matrix to a 3-tuple
    ##### INPUTS #####
    :matrix           A sparse matrix. E.g np.matrix or sp.csr or ...
    ##### RETURN #####
    :tuple            A 3-tuple (indices, values, shape)
    """
    if (type(matrix) == np.matrix) or (type(matrix) == np.array):
        matrix = matrix.tocoo()
    if (not type(matrix) == sp.coo):
        matrix = matrix.tocoo()        
    tuple3 = (zip(matrix.row, matrix.col), matrix.data, matrix.shape)
    return tuple3
"""
##### INPUTS #####
:adj_normalized      The preprocessed adj: = D^-0.5 * (adj+I) * D^-0.5
:adj                 sparse represenation of (adj + I) = (coord, values, shape)
:features            sparse represenation of the feature matrix X_0 = (coord, values, shape)
:placeholders        A Dictionary: {'features'[Nxm],'adj'[NxN], 'adj_orig'[NxN], 'dropout'(float), 'real_distribution'[Nx32]}
##### RETURN #####
:feed_dict:          A Dictionary: {placeholders['features']: features, placeholders['adj']:adj_norm, placeholders['adj_orig']: adj}
"""
def construct_feed_dict(features, labels, dropout, placeholders):
    # check types
    if (not type(features) == tuple):
        features = dense_to_sparse(features)
    if (not type(labels) == tuple):
        labels = dense_to_sparse(labels)
    
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['dropout']: dropout})
    return feed_dict

"""
Function to build test set with X% positive links
NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
##### INPUTS #####
:labels                The ajacency matrix of a graph       
:tetd                  A dictionary: <row-in-matrix> ==> 'test' / 'train' / 'eval'
##### RETURN #####
:train:                The training labels
:val:                  The evaluation labels
:val_false:            Some false evaluation labels
:test:                 The test labels
:test_false:           Some false test labels
:train_mask            An array of (True / False) values and length N.
                       It is true at pos X, iff the node at pos. X belongs to the training data
:val_mask              -||-
:test_mask             -||-
"""
def make_train_eval_test(labels, tetd):
    
    N = labels.shape[0]
    m = labels.shape[1]
    
    d = collections.Counter(tetd.values())
    num_train = d['train']
    num_eval = d['eval']
    num_test = d['test']
    
    #create dummies
    train = sp.lil_matrix((num_train, m))
    val = sp.lil_matrix((num_eval, m))
    val_false = sp.lil_matrix((num_eval, m))
    test = sp.lil_matrix((num_test, m))
    test_false = sp.lil_matrix((num_test, m))
    
    num_train = 0
    num_eval = 0
    num_test = 0
    
    train_mask = [False] * N
    val_mask = [False] * N
    test_mask = [False] * N
    
    # divide all the edegs from adj into test and training based on the tetd
    for idx in range(N):
        row = labels[idx]
        
        if tetd[idx] == 'train':
            train[num_train] = row
            num_train += 1
            train_mask[idx] = True
            
        elif tetd[idx] == 'eval':
            val[num_eval] = row
            num_eval += 1
            val_mask[idx] = True
            
        elif tetd[idx] == 'test':
            test[num_test] = row
            num_test += 1
            test_mask[idx] = True
        else:
            print __file__, " unknown: ", tetd[row]
    
    print "Loaded {}(Train)<==>{}(Eval)<==>{}(Test)".format(d['train'], d['eval'], d['test'])
    
    # define an array and fill it with random positions \in [0, m]
    # avg_nonzero_entries_in_row_of_label = X
    X = int(np.mean(np.sum(labels, axis=1)))    
    def get_rand_pos():
        rand_pos = []    
        for _ in range(X):
            rand_pos.append(random.randint(0, m-1))
        return rand_pos
    
    for i in range(num_eval):
        rand_pos = get_rand_pos()
        row = sp.lil_matrix((1, m))
        for p in rand_pos:
            row[0, p] = 1
        val_false[i] = row
        
    for i in range(num_test):
        rand_pos = get_rand_pos()
        row = sp.lil_matrix((1, m))
        for p in rand_pos:
            row[0, p] = 1
        test_false[i] = row
    
    return train, val, val_false, test, test_false, train_mask, val_mask, test_mask
