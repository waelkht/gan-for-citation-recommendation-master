import collections
import cPickle as pkl
import scipy.sparse as sp
import numpy as np
import os
from shutil import copyfile
from arga import Z_recommend

"""
A helper function which sorts the embeddings and the adj_orig
##### INPUTS #####
:embeddings         The embeddings from the arga model (matrix of size [N x m])
:adj_orig           The original adjacency matrix of size [N x N]
:reconstructions    The matrix A' from the arga model of size [N x N]
:tetd               A dictionary <row-index> ==> ('train' / 'eval' / 'test')
##### RETURN #####
:train_and_eval_rec The reconstructions which have the label 'train' or 'eval' in tetd
:test_rec           The reconstructions which have the label 'test' in tetd
:train_and_eval_lab The labels which have the label 'train' or 'eval' in tetd
:test_labels        The labels which have the label 'test' in tetd
:nnew_adj           Since the rows in the embeddings swapped places, we have to do
                      this also in the adj. matrix and the rec. matrix
"""
def helper_sort(rec, orig_features, adj_orig, tetd):
    occurrences = collections.Counter(tetd.values())
    num_train = occurrences['train']
    num_eval = occurrences['eval']
    num_test = occurrences['test']
    assert( (num_train + num_eval + num_test) == len(tetd))
    assert( rec.shape[0] == orig_features.shape[0])
    
    vector_size = rec.shape[1]
    
    train_rec = sp.lil_matrix((num_train, vector_size))
    eval_rec = sp.lil_matrix((num_eval, vector_size))
    test_rec = sp.lil_matrix((num_test, vector_size))    
    
    train_feature = sp.lil_matrix((num_train, vector_size))
    eval_feature = sp.lil_matrix((num_eval, vector_size))
    test_feature = sp.lil_matrix((num_test, vector_size))
    
    if not type(adj_orig) == np.array:
        adj_orig = adj_orig.todense()
    if not type(orig_features) == np.array:
        orig_features = orig_features.todense()
    new_adj = np.zeros(adj_orig.shape)
    
    idx_train = 0
    idx_eval = 0
    idx_test = 0
    new_indices = dict()
    
    for num_row in range(len(tetd)):
        row_rec = rec[num_row, :]
        row_feature = orig_features[num_row, :]
        if tetd[num_row] == 'train':
            train_rec[idx_train, :] = row_rec
            train_feature[idx_train, :] = row_feature
            new_indices[num_row] = idx_train
            idx_train += 1
        elif tetd[num_row] == 'eval':
            eval_rec[idx_eval, :] = row_rec
            eval_feature[idx_eval, :] = row_feature
            new_indices[num_row] = idx_eval + num_train
            idx_eval += 1         
        elif tetd[num_row] == 'test':
            test_rec[idx_test, :] = row_rec
            test_feature[idx_eval, :] = row_feature
            new_indices[num_row] = num_train + num_eval + idx_test
            idx_test += 1
        else:
            print __file__, " unknown ", tetd[num_row]
    
    del rec
    del orig_features
    
    # swap rows
    for old, new in new_indices.items():
        new_adj[new, :] = adj_orig[old, :]
    del adj_orig
            
    nnew_adj = np.zeros(new_adj.shape)
    
    # swap columns
    for old, new in new_indices.items():
        nnew_adj[:, new] = new_adj[:, old]
    del new_adj
    
    train_and_eval_rec = sp.vstack((train_rec, eval_rec))
    train_and_eval_features = sp.vstack((train_feature, eval_feature))
    
    return sp.csr_matrix(train_and_eval_rec), sp.csr_matrix(test_rec), sp.csr_matrix(train_and_eval_features), sp.csr_matrix(test_feature), sp.csr_matrix(nnew_adj)

"""
This function will drop dump all the data which will be used for evaluation later.
##### INPUTS #####
:reconstructions    The reconstructions from the arga2 model
:orig_features      The original features of the nodes
:data_name          The name: e.g. 'aan/paper_vec/paper_title_vec'
:model              The model: e.g 'arga_ae'
"""
def dump(reconstructions, orig_features, data_name, model):
    
    print "[INFO] Dumping results ..."
    
    # Load the test-eval-train-dict and the adj. matrix
    tetd_name = "../data/{}.train_eval_test_dict.pkl".format(data_name)
    adj_name = "../data/{}.adj.pkl".format(data_name)
    tetd = pkl.load(open(tetd_name))
    adj = pkl.load(open(adj_name))
    
    # sort the embeddings into two chunks: [train, eval] <==> [test]
    # The adj matrix changes accordingly   
    train_and_eval_rec, test_rec, train_and_eval_features, test_features, adj_sort = helper_sort(reconstructions, orig_features, adj, tetd)
    
    # dump [train, eval] + [test] + sorted_adj_orig
    result_name = "result/{}.{{}}.{}".format(data_name, model)
    pkl.dump(train_and_eval_features, open(result_name.format("train_eval_features"), 'w'))
    pkl.dump(test_features, open(result_name.format("test_features"), 'w'))
    pkl.dump(train_and_eval_rec, open(result_name.format("train_eval_reconstruct"), 'w'))
    pkl.dump(test_rec, open(result_name.format("test_reconstruct"), 'w'))
    pkl.dump(adj_sort, open(result_name.format("adj_sort"), 'w'))
    
    print "[INFO] Dumping results finished"
    
    return train_and_eval_rec, test_rec, train_and_eval_features, test_features, adj_sort

"""
This function takes the outputs from one or more arga runs, and joins them together.
##### INPUTS #####
:results_list   A list of datasets which we want to join together
:model          The name of the model. E.g. "arga_ae"
:result_file    The name of the file where the result will be dumped. E.g. "aan/merged/merged"
"""
def join_results(results_list, model, result_file):
    assert(len(results_list) > 0)
    
    path_train_eval_embeddings = "result/{{}}.train_eval_embeddings.{}".format(model)
    path_test_embeddings = "result/{{}}.test_embeddings.{}".format(model)
    
    # load the first train_eval and test embeddings...
    train_eval_embeddings = pkl.load(open(path_train_eval_embeddings.format(results_list[0])))
    test_embeddings = pkl.load(open(path_test_embeddings.format(results_list[0])))
    
    # join the other horizontally
    for i in range(1, len(results_list)):
        additional_train_eval_embeddings = pkl.load(open(path_train_eval_embeddings.format(results_list[i])))
        additional_test_embeddings = pkl.load(open(path_test_embeddings.format(results_list[i])))
        train_eval_embeddings = sp.hstack((train_eval_embeddings, additional_train_eval_embeddings))
        test_embeddings = sp.hstack((test_embeddings, additional_test_embeddings))
        
    # copy the train_eval_test_dict
    if not os.path.exists(result_file):
        os.makedirs("result/{}".format(result_file))
        os.rmdir("result/{}".format(result_file)) # delete not the whole path, but only the last dir
        os.makedirs("data/{}".format(result_file))
        os.rmdir("data/{}".format(result_file)) # delete not the whole path, but only the last dir
        
    copyfile("result/{}.adj_sort.{}".format(results_list[0], model),
             "result/{}.adj_sort.{}".format(result_file, model))
    copyfile("data/{}.train_eval_test_dict.pkl".format(results_list[0]),
             "data/{}.train_eval_test_dict.pkl".format(result_file))
    copyfile("data/{}.adj.pkl".format(results_list[0]),
             "data/{}.adj.pkl".format(result_file))
    
    # dump the joined matrices:
    pkl.dump(train_eval_embeddings, open(path_train_eval_embeddings.format(result_file), 'w'))
    pkl.dump(test_embeddings, open(path_test_embeddings.format(result_file), 'w'))
    
"""
It is sometimes necessary, to evaluate directly on the features without processing it with the arga model.
This is in example necessary for cosine similarity.
This function turns a feature matrix and a adj. matrix in a result format.
##### INPUTS #####
:dataset        The name of the dataset. e.g. "aan/paper_abstract_vec/paper_abstract_vec"
:model          The name which the result will have: "aan/paper_abstract_vec/paper_abstract_vec.test_embeddings.<model>"
:result_file    The name of the result: E.g. "aan/abtract_cosine/abstract_cosine"
""" 
def feature_to_result(dataset, model, result_file):
    data_path = "data/{}.{}.pkl"
    adj_path = data_path.format(dataset, "adj")
    feature_path = data_path.format(dataset, "features")
    tetd_path = data_path.format(dataset, "train_eval_test_dict")
    
    adj_matrix = pkl.load(open(adj_path)) 
    features = pkl.load(open(feature_path))
    tetd = pkl.load(open(tetd_path))
    
    train_test_embeddings, test_embeddings, adj_sort = helper_sort(features, adj_matrix, tetd)
    
    # copy the train_eval_test_dict
    if not os.path.exists(result_file):
        os.makedirs("result/{}".format(result_file))
        os.rmdir("result/{}".format(result_file)) # delete not the whole path, but only the last dir
    
    result_path = "result/{{}}.{{}}.{}".format(model)
    pkl.dump(train_test_embeddings, open(result_path.format(result_file, "train_eval_embeddings"), 'w'))
    pkl.dump(test_embeddings, open(result_path.format(result_file, "test_embeddings"), 'w'))
    pkl.dump(adj_sort, open(result_path.format(result_file, "adj_sort"), 'w'))
    print "[INFO] Converted feature to result:"
    print "       data/{} ==> result/{}".format(dataset, result_file)

def test_sort():
    emb = np.array([[0, 1],
                  [2, 3],
                  [4, 5],
                  [6, 7],
                  [8, 9]])
    adj = sp.csr_matrix([[0, 0, 1, 0, 1],
                         [1, 0, 0, 1, 0],
                         [0, 1, 0, 0, 1],
                         [1, 0, 1, 0, 0],
                         [1, 1, 0, 1, 1]])
    tetd = dict({0: 'train', 1: 'test', 2: 'eval', 3: 'test', 4: 'eval'})
    
    train_and_eval_embeddings, test_emb, adj_sort = helper_sort(emb, adj, tetd)
    
#     [[0, 0, 1, 0, 1],
#      [0, 1, 0, 0, 1],
#      [1, 1, 0, 1, 1],
#      [1, 0, 0, 1, 0],
#      [1, 0, 1, 0, 0]])

    g = np.array([[0, 1, 1, 0, 0],
                  [0, 0, 1, 1, 0],
                  [1, 0, 1, 1, 1],
                  [1, 0, 0, 0, 1],
                  [1, 1, 0, 0, 0]], dtype=np.float64)
    g2 = np.array([[0, 1],
                  [4, 5],
                  [8, 9],
                  [2, 3],
                  [6, 7]])

    r = np.sum(adj_sort - g) == 0
    r2 = np.sum(sp.vstack((train_and_eval_embeddings, test_emb)) - g2) == 0
    if not (r and r2):
        print "Sort:"
        print adj_sort.todense()
        print "Goal: "
        print g
    return r    

if __name__ == "__main__":
    
    if not test_sort():
        exit("test failed")
    else:
        print "Test passed"
    
#     emb = pkl.load(open("tmp/emb", 'r'))
#     adj = pkl.load(open("tmp/adj", 'r'))
#     dump(emb, adj, "aan/paper_vec/paper_title_vec", "arga_ae")
#     results = ["aan/paper_author/paper_author", "aan/paper_title_vec/paper_title_vec"]
#     join_results(results, "arga_ae", "aan/merged/merged")

    feature_to_result("aan/paper_title_vec/paper_title_vec", "arga_ae", "aan/title_cosine/title_cosine")
    
    Z_recommend.perf("aan/paper_title_vec/paper_title_vec", "arga_ae")
    
# MERGED:
#     Recall@20 = 0.10594652832
#     Recall@40 = 0.171595202613
#     Recall@60 = 0.221297544158
#     Recall@80 = 0.267026888567
#     Recall@100 = 0.306125508653
