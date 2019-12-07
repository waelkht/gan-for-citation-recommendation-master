import cPickle as pkl
import scipy.sparse as sp
import tensorflow as tf
import numpy as np
from arga.output_data import helper_sort
from sklearn.metrics.pairwise import cosine_similarity
from arga.Z_recommend import output_performance


def load_data(result_path, history_path, other_features, model, tetd_path):
    """
    This function loads the data from the file system, prepares it and returns it.
    ##### INPUTS #####
    :result_path            The path to the result files: (adj_sort, test_embeddings, train_eval embeddings)
    :history_path           The path to the history file in the 'Dataset/...' directory
    :other_features         A list of feature files. The cosine sim. is taken from it and also returned
    :model                  The name of the model. E.g 'arga_ae'
    :tetd_path              The name of the path to the train_eval_test_dict. 
    ##### OUTPUT #####
    :train_eval_sim         The embeddings of the train and eval data
    :test_sim               The embeddings from the test data
    :train_eval_hist        The history of the train and eval data in form of a list
    :test_hist              The history of the test data in form of a list
    :train_eval_adj_sort    The train and eval data of the sorted adj. matrix
    :test_adj_sort          The test data of the sorted adj. matrix
    """

    print "[INFO] Loading data ..."

    train_eval_embeddings_name = "{}.train_eval_embeddings.{}".format(result_path, model)
    test_embeddings_name = "{}.test_embeddings.{}".format(result_path, model)
    adj_sort_name = "{}.adj_sort.{}".format(result_path, model)

    with open(train_eval_embeddings_name) as train_eval_embeddings_file:
        train_eval_embeddings = pkl.load(train_eval_embeddings_file)        

    with open(test_embeddings_name) as test_embeddings_file:
        test_embeddings = pkl.load(test_embeddings_file)

    with open(adj_sort_name) as adj_sort_file:
        adj_sort = pkl.load(adj_sort_file)

    with open(tetd_path) as tetd_file:
        tetd = pkl.load(tetd_file)

    with open(history_path) as history_file:
        history = pkl.load(history_file)

    number_train_eval = train_eval_embeddings.shape[0]
    number_test = test_embeddings.shape[0]

    train_eval_sim = cosine_similarity(train_eval_embeddings)
    test_sim = cosine_similarity(test_embeddings, train_eval_embeddings)

    train_eval_other = []
    test_other = []

    # First we add the other features...
    for path in other_features:
        with open(path) as feature_file:
            features = pkl.load(feature_file)
        N = features.shape[0]
        fake_adj_matrix = sp.lil_matrix((N, N))
        train_eval_features, test_features, _ = helper_sort(features, fake_adj_matrix, tetd)
        _train_eval_other = cosine_similarity(train_eval_features, train_eval_features)
        _test_other = cosine_similarity(test_features, train_eval_features)
        _train_eval_other = sp.csr_matrix(_train_eval_other)
        _test_other = sp.csr_matrix(_test_other)
        train_eval_other.append(_train_eval_other)
        test_other.append(_test_other)


    # ... then we sort the history matrices
    for history_matrix in history:
        N = history_matrix.shape[0]
        fake_embeddings = sp.lil_matrix((N, 10)) # 10 is just any random value, it doesn't matter

        _, _, hist_sort = helper_sort(fake_embeddings, history_matrix, tetd)
        _train_eval_other = hist_sort[:number_train_eval,  :number_train_eval]
        _test_other       = hist_sort[ number_train_eval:, :number_train_eval]
        train_eval_other.append(_train_eval_other)
        test_other.append(_test_other)


    train_eval_adj_sort = adj_sort[:number_train_eval , :number_train_eval]
    test_adj_sort       = adj_sort[ number_train_eval:, :number_train_eval]

    print "[INFO] Loading data finished."

    return train_eval_sim, test_sim, train_eval_other, test_other, train_eval_adj_sort, test_adj_sort

def train_model(train_eval_sim, train_eval_hist, train_eval_adj_sort, num_iterations):
    """
    This function trains the model and returns the weights such that the function is minimized:
    error = | (w_1 * SIM + w_2 * A1 + w_3 * A2 + ... ) - ADJ |
    ##### INPUTS #####
    :train_eval_sim         The embeddings of the train and eval data
    :test_sim               The embeddings from the test data
    :train_eval_hist        The history of the train and eval data in form of a list
    :test_hist              The history of the test data in form of a list
    :train_eval_adj_sort    The train and eval data of the sorted adj. matrix
    :test_adj_sort          The test data of the sorted adj. matrix
    ##### OUTPUT #####
    :weights                A list of weights for the minimal error of the function above
    """
    print "[INFO] Training model ..."
    
    num_weight = len(train_eval_hist) + 1
    
    # create the weights
    weights = []
    example = 0.5
    for i in range(num_weight):
        weights.append(tf.Variable(example, dtype=tf.float32)) # 0.1 is just a random value
    
    # create the matrices
    matrices = []
    matrices.append(tf.constant(train_eval_sim, dtype=tf.float32))
    for history in train_eval_hist:
        matrices.append(tf.constant(history.todense(), dtype=tf.float32))
        
    # define the loss:
    # _sum = (w_1 * SIM + w_2 * A1 + w_3 * A2 + ... ) - ADJ
    _adj = tf.constant(train_eval_adj_sort.todense(), dtype=tf.float32)
    _sum = tf.add_n([w * M for w, M in zip(weights, matrices)])
    _dif = _sum - _adj
    # loss = | _sum |
    loss_total = tf.reduce_sum(tf.abs(_dif))
    loss_one = tf.reduce_sum(tf.abs(tf.multiply(_dif, _adj)))
    loss_zero = loss_total - loss_one
    num_ones = tf.reduce_sum(_adj)
    num_total = int(_adj.shape[0] * _adj.shape[1])
    weighted_loss = (num_ones * loss_zero) + ((num_total - num_ones) * loss_one) 
    
    # define optimizer
    optimizer = tf.train.AdamOptimizer(0.01)
    step = optimizer.minimize(weighted_loss)
    
    init = tf.initialize_all_variables() 
    
    with tf.Session() as sess:
        # init
        sess.run(init)
        
        # train model
        for i in range(num_iterations):
            _loss, _ = sess.run([weighted_loss, step])
            returnable_weights = sess.run(weights)
            print "[INFO] Loss in iteration {} = {}".format(i, _loss)
        
    print "[INFO] Training model finished."
    
    return returnable_weights

def run(result_path, history_path, other_features, model, tetd_path):
    train_eval_sim, test_sim, train_eval_other, test_other, train_eval_adj_sort, test_adj_sort = load_data(result_path, history_path, other_features, model, tetd_path)

    print "Before"
    output_performance(test_sim, test_adj_sort)

    def get_recall(_weights):
        print "After"
        _new_test_sim = _weights[0] * test_sim
        for w, M in zip(_weights[1:], test_other):
            _new_test_sim = np.add(_new_test_sim, w * np.asarray(M.todense()))
        output_performance(_new_test_sim, test_adj_sort)
        
    weights = train_model(train_eval_sim, train_eval_other, train_eval_adj_sort, 100)

    get_recall(weights)
    print "Weights: {}".format(weights)
    print "[INFO] finished"


if __name__ == '__main__':
    result_path = "../arga/result/aan/joined/joined"
    history_path = "../Datasets/AAN/processed/history.pkl"
    paper_abstract_vec = "../data/aan/paper_abstract_vec/paper_abstract_vec.features.pkl"
    model = "arga_ae"
    tetd_path = "../Datasets/AAN/processed/train_eval_test_dict.pkl"

    run(result_path, history_path, [paper_abstract_vec], model, tetd_path)
