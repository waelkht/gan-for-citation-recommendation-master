import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os.path

# open a .index file and get all the numbers in it...
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# Creates an bool-array of length l, where the indices of given in idx are true
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

"""
This function loads the adjacency matrix and the feature matrix from "data/{dataset}.adj.pkl"
and "data/{dataset}.features.pkl" if possible. Otherwise it uses {x, y, tx, ty, allx, ally}...
##### INPUTS #####
:dataset      The name of the dataset. E.g. 'cora'
##### RETURN #####
:adj:         The Adjacency-matrix of the graph [N x N] where N is the number of nodes. Type: scipy.sparse.csr.csr_matrix
:features:    the features of the graph: [N x m] where m is the number of features (...101001...). Type: scipy.sparse.csr.csr_matrix
"""
def load_data(dataset):
    filename_adj = "../data/{}.adj.pkl".format(dataset)
    filename_feature = "../data/{}.features.pkl".format(dataset)
    filename_tetd = "../data/{}.train_eval_test_dict.pkl".format(dataset)
    
    if os.path.isfile(filename_adj) and os.path.isfile(filename_feature):# and os.path.isfile(filename_ttd):
        adj = pkl.load(open(filename_adj, 'r'))
        features = pkl.load(open(filename_feature, 'r'))
        train_eval_test_dict = pkl.load(open(filename_tetd, 'r'))
    else:
        adj, features = load_data_casual(dataset)
        
    return adj, features, train_eval_test_dict

"""
Loads the data of one the examples in data/...
#### INPUT #####
:dataset:    a String. eg. 'cora'
# RETURN
:adj:        The Adjacency-matrix of the graph [N x N] where N is the number of nodes. Type: scipy.sparse.csr.csr_matrix
:features:   the features of the graph: [N x m] where m is the number of features (...101001...). Type: scipy.sparse.csr.csr_matrix
"""
def load_data_casual(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    # load the objects via pickle:
    # x: [ 140 x 1433 ] feature-vecs for training of nodes which also have a label y
    # tx: [ 1000 x 1433 ] feature-vecs for testing of nodes which also have a label ty
    # allx: [1708 x 1433] feature-vecs of all nodes. allx is superset of x
    # y: [ 140 x 7] label-vecs for training of nodes in x
    # ty: [ 1000 x 7] label vecs for nodes in tx
    # ally: [ 1708 x 7] label-vecs for all nodes in allx
    # graph: a graph
    for i in range(len(names)):
        objects.append(pkl.load(open("../data/ind.{}.{}".format(dataset, names[i]))))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset)) # get all the numbers in .index
    test_idx_range = np.sort(test_idx_reorder) # sort the numbers of .index, text_idx_range = [1708:2708]

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil() # [ 2708 x 1433 ]
    features[test_idx_reorder, :] = features[test_idx_range, :] # reorder the lower part of the features...
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)) # [ 2708 x 2708 ]

    return adj, features.tocsr()

# the same as load_data(dataset)... with more return variables:
# RETURN
# y_train: label-vector for the training data [N x k] the labels at train_mask are set
# y_val: label-vector for the training data [N x k] the labels at val_mask are set
# train_mask: the mask which indicates where the labels of y_train are set [N]. It consists of bools
# val_mask: the mask which indicates where the labels of y_val are set [N]. It consists of bools
def load_alldata(dataset_str):
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset_str, names[i]))))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, np.argmax(labels, 1)
