import pickle as pkl
import os.path

def load_data(dataset):
    
    """
    This function loads the adjacency matrix and the feature matrix from "data/{dataset}.adj.pkl"
    and "data/{dataset}.features.pkl" if possible. Otherwise it uses {x, y, tx, ty, allx, ally}...
    ##### INPUTS #####
    :dataset      The name of the dataset. E.g. 'cora'
    ##### RETURN #####
    :label        The labeled data: [N x n]
    :features:    the features of the graph: [N x m] where m is the number of features (...101001...). Type: scipy.sparse.csr.csr_matrix
    :tetd         The train_eval_test dict: <row> ==> 'train' / 'eval' / 'test'
    """
    
    filename_label = "../data/{}.label.pkl".format(dataset)
    filename_feature = "../data/{}.features.pkl".format(dataset)
    filename_tetd = "../data/{}.train_eval_test_dict.pkl".format(dataset)
    filename_divi = "../data/{}.divisions.pkl".format(dataset)
    
    assert(os.path.isfile(filename_label))
    assert(os.path.isfile(filename_feature))
    assert (os.path.isfile(filename_tetd))
    
    label = pkl.load(open(filename_label, 'r'))
    features = pkl.load(open(filename_feature, 'r'))
    train_eval_test_dict = pkl.load(open(filename_tetd, 'r'))
    
    try:
        divisions = pkl.load(open(filename_divi))
    except:
        divisions = [features.shape[1]]
        print "No division of feature matrix provided. Using [{}] instead.".format(divisions[0])
        
    return label, features, train_eval_test_dict, divisions
