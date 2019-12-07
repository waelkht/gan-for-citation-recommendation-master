import numpy as np
import pickle as pkl
import random
import heapq
import operator
import sys
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import output_data
from arga import Z_recommend


def get_real_indices(m):
    """
    This function calculates the citations from the adjacency matrix
    ##### INPUTs #####
    :m          The adjacency matrix (sparse or numpy matrix)
    ##### RETURN #####
    :citations  A list of lists representing the citations.
    Example: [[0, 0, 1], [1, 0, 1]] ==> [[2], [0, 2]] 
    """
    N = m.shape[0]
    citations = list()
    for i in range(N):
        citations.append(m[i].tocoo().col)
    return citations

def _get_top_n_indices(n, iterabl):
    final_list = zip(*heapq.nlargest(n, enumerate(iterabl), key=operator.itemgetter(1)))[0]
    return final_list
        
def get_most_top_n_indices(sim, n=100):
    """
    This function calculates the indices of the highest entries of a matrix for each row.
    ##### INPUT #####
    :sim_mat       A numpy array or matrix of the similarities
    ##### RETURN #####
    :sim_indices  The indicies of the most similar documents
    Example: [[0.2, 0.7, 0.1], [0.5, 0.3, 0.9]] ==> [[1, 0, 2], [2, 0, 1]]    
    """
    # convert matrix if necessary
    if (not type(sim) == np.ndarray) and (not type(sim) == np.matrix):
        sim = sim.toarray()
    if type(sim) == np.matrix:
        sim = np.array(sim)
        
    sim_indices = list()
    for i in range(sim.shape[0]):
        sim_indices.append(_get_top_n_indices(n, sim[i]))
        
    return sim_indices

def _load_data(dataset, model):
    """
    Loads the train_eval embeddings, the test embeddings and the original adj. matrix
    ##### INPUTS #####
    :dataset        The name of the dataset. E.g. "aan/joined/joined"
    :model          The name of the model. E.g. "arga_ae"
    ##### RETURNS #####
    :train_eval_emb The train and eval embeddings
    :test_emb       The test embeddings
    :orig           The original adj. matrix
    """
    train_eval_rec = pkl.load(open('result/{}.train_eval_reconstruct.{}'.format(dataset, model), 'r')) # scipy sparse matrix
    test_rec = pkl.load(open('result/{}.test_reconstruct.{}'.format(dataset, model), 'r')) # scipy sparse matrix
    train_eval_features = pkl.load(open('result/{}.train_eval_features.{}'.format(dataset, model), 'r')) # scipy sparse matrix
    test_features = pkl.load(open('result/{}.test_features.{}'.format(dataset, model), 'r')) # scipy sparse matrix
    
    orig = pkl.load(open('result/{}.adj_sort.{}'.format(dataset, model), 'r')) # scipy sparse matrix
    try:   
        divi = pkl.load(open('../data/{}.divisions.pkl'.format(dataset), 'r')) # list  
    except:
        divi = [test_features.shape[1]] # could also be train_eval_rec ...
    
    return train_eval_rec, test_rec, train_eval_features, test_features, orig, divi

def divide_features(features, divisions):
    """
    This function divides the feature matrix into multiple matrices according to divisions.
    ##### INPUTS #####
    :features     A matrix of the features [N x ?]
    :divisions    A list of integers, indicating the sizes of the individual matrices
    ##### OUTPUT #####
    :list         A list of matrices
    """
    
    assert(features.shape[1] == sum(divisions))
    
    result = []
    
    idx = 0
    for size in divisions:
        sub_mat = features[:, idx : idx+size]
        result.append(sub_mat)
        idx += size
        
    return result

def get_top_n_popositions(matrices, tops=None):
    """
    This function will calculate the top n propositions in the matrices
    ##### INPUTS #####
    :matrices           A list of matrices
    :tops               A list of integers. An integer indicates how many top positions shall be proposed
    ##### OUTPUT #####
    :list               A list of top positions for each row in each matrix.
    """
    # if no tops is given, we take the top 10%
    if tops == None:
        tops = []
        for M in matrices:
            tops.append(M.shape[1] / 10)
            
    assert(len(matrices) == len(tops))
    
    result = []
    # get the top n in each matirx
    for n, M in zip(tops, matrices):
        _tmp = get_most_top_n_indices(M, n)
        result.append(_tmp)
        
    return result

def get_real_propositions(matrices):
    """
    Get the entries for the real features
    """
    
    _result = []
    for M in matrices:
        _result.append(get_real_indices(M))
        
    return _result

def get_intersection(propositions, real_features, num_features, filt):
    """
    This function calculates the intersections of the proposition matrix and the
    real_feature matrix via a matrix multiplication.
    TODO: filt is not used yet
    """
    
    height = len(propositions)
    width = len(real_features)
        
    A = sp.lil_matrix((height, num_features))
    B = sp.lil_matrix((width, num_features))
    
    def fill_matrix(matrix, llist):
        for i, row in enumerate(llist):
            matrix[i, row] = 1
    
    # convert propositions and real_features back to matrices
    fill_matrix(A, propositions)
    fill_matrix(B, real_features)
    
    res = np.multiply(A, np.transpose(B)).toarray()
    res = np.clip(res, 0, 1)
    
            
    return res

def get_intersections(propositions, real_features, num_features, filt=None):
    """
    This function calculates multiple intersections matrices
    ##### INPUTS #####
    :propositions     A list of matrices(in array format) with proposed features [[N x m1], [N x m2], ...]
    :real_featurs     A list of matrices(in array format) with the real features [[M x ?], [M x ?], ...]
    :filter           A list of integers or floats, indicating the minimum number of similar entries
                      or the minimal recall.
    ##### OUTPUT #####
    :list              A list of intersection matrices [[N x M], [N x M], ...]
    """
    
    if filt == None:
        filt = [1]*len(propositions)
    
    assert(len(propositions) == len(real_features))
    assert(len(propositions) == len(filt))
    
    _res = []
    
    for X, Y, f in zip(propositions, real_features, filt):
        _res.append(get_intersection(X, Y, num_features, f))
    
    return _res
    
def get_cosine_matrix(path_vecs, path_tetd):
    
    """
    This function calculates the cosine similarity for the train-eval vecs and the test vecs
    ##### INPUTS #####
    :path_vecs      The path to the vectors of abstract or title
    :path_tetd      The path to the train_eval_test_dict
    ##### RETRUN #####
    :cosine         The cosine similarity between the test vecs and the train+eval vecs
    """
    
    with open(path_vecs) as input_file:
        vecs = pkl.load(input_file)
    with open(path_tetd) as input_file:
        tetd = pkl.load(input_file)
    
    num_nodes = vecs.shape[0]
    vector_length = vecs.shape[1]
    dummy_adj = sp.csr_matrix((num_nodes, num_nodes))
    dummy_rec = sp.csr_matrix((num_nodes, vector_length))
    
    train_eval_vecs, test_vecs, _, _, _ = output_data.helper_sort(vecs, dummy_rec, dummy_adj, tetd)
    
    all_vecs = sp.vstack((train_eval_vecs, test_vecs))
    cosine = cosine_similarity(test_vecs, all_vecs)
    
    return cosine
        
def sort_combination(combination, cosine):
    """
    This function sorts the matrices in intersections according to the cosine similarity.
    ##### INPUTS #####
    :intersections        An list of matrices. All matrices have to have the same shape as cosine
    :cosine               A cosine similarity matrix between the test set and the train eval set
    ##### RETURN #####
    :intersections        An list of lists of lists, indicating the best similarities for each row in each matrix.                          
    """
    
    assert(combination.shape == cosine.shape)        
    
    _comb = np.multiply(combination, cosine) # watch out! this is element wise, not a matrix multiplication!
    res = get_most_top_n_indices(_comb)
        
    return res

def output_MRR(list_of_citations, sim_indices):
    """
    This function calculates the Mean Reciprocal Rank (MRR)
    """
    N = len(list_of_citations)
    
    _sum = 0.0
    for i in range(N):
        real_citations = list_of_citations[i]
        pred_citations = sim_indices[i]
        for k, citation in enumerate(pred_citations):
            if citation in real_citations:
                _sum += (1.0 / (k+1))
                break
            
    _sum /= N
    
    print "MRR = {}".format(_sum)
            
def output_MAP(list_of_citations, sim_indices):
    N = len(list_of_citations)
    
    _MAP = 0.0
    for i in range(N):
        real_citations = list_of_citations[i]
        if len(real_citations) > 0:
            pred_citations = sim_indices[i]
            _AP = 0.0 # average precision
            _cntr = 1.0 # count how many correct citations have been made already
            for k, recommendation in enumerate(pred_citations):
                if recommendation in real_citations:
                    _AP += (_cntr / (k+1))
                    _cntr += 1
            _AP /= len(real_citations)
            _MAP += _AP
    _MAP /= N       
    
    print "MAP = {}".format(_MAP)

def get_recall(best_indices, adj_sort,):
    
    """
    This function calculates the recall for 20, 40, 60, 80 and 100
    ##### INPUTS #####
    :best_indices          The best indices. A list of lists
    :adj_sort              The adj. matrix of the real citations that have been made
    """
    
    num_test = len(best_indices)
    num_train_eval = adj_sort.shape[0] - num_test
    
    test_citations = adj_sort[num_train_eval:, :]    
    real_citations = get_real_indices(test_citations)
    
    output_MAP(best_indices, real_citations)
    output_MRR(best_indices, real_citations)
    
    
    for recall in [20, 40, 60, 80, 100]:
        score = 0.
        for row_idx in range(num_test):
            y_real = set(real_citations[row_idx])
            if len(y_real) > 0:
                y_pred = set(best_indices[row_idx][0:recall])
                common = y_pred.intersection(y_real)
                _rec = (1.*len(common)) / len(y_real)
                score += _rec
        score /= num_test
        print "Recall@{}: {}".format(recall, score)
    
def combine_intersections(best_indices):
    
    fin = np.zeros(best_indices[0].shape)
    for matrix in best_indices:
        fin += matrix
    
    return fin

def tickle(pred_features):
    """
    This function substracts the average of each column.
    ##### INPUTS #####
    :pred_features    A list of matrices from the model.
    ##### OUTPUTS #####
    :pred_features    A list of matrices, where each matrix has been subtracted by the average of the column.
    """
    
    for i, M in enumerate(pred_features):
        _avg = np.average(M.toarray(), axis=0)
        _sub = np.subtract(M.toarray(), _avg)
        pred_features[i] = sp.csr_matrix(_sub)
        
    return pred_features
        

"""
This function will calculate the performance of the citation.
For every paper it uses the cos-sim of the embeddings to find similar papers and than prints the recall.
##### INPUTS ####
:dataset        The name of the dataset. e.g 'cora'
:model          The name of the model e.g. 'arga_ae'
:divisions      The divisions of the feature matrix.
                E.g. [5123, 376] indicating the first 5123 columns belong to the authors etc.
:train_eval_emb The train and eval embeddings
:test_emb       The test embeddings
:orig           The original adj. matrix
"""
def perf(dataset, model, divisions=None, train_eval_rec=None, test_rec=None, train_eval_features=None, test_features=None, adj_sort=None):
    print "[INFO] Start calculating recall ..." 
    
    # if the matrices are not provided ==> load them
    if train_eval_rec==None or test_rec==None or train_eval_features==None or test_features == None or adj_sort==None or divisions==None:
        train_eval_rec, test_rec, train_eval_features, test_features, adj_sort, divisions = _load_data(dataset, model)
    
    # Some features are only proposed, since they occur often. We have to remove this effect via the mean.
    # MEAN = np.mean(train_eval_rec, axis = 0)
    # train_eval_rec -= MEAN
    # test_rec -= MEAN

    num_features = train_eval_features.shape[1]
    num_train_eval = train_eval_rec.shape[0]
    num_test = test_rec.shape[0]

    # 1. divide the feature C' matrix into list of features L'
    pred_features = divide_features(test_rec, divisions)
    pred_features = tickle(pred_features)
    
     
    # 2. divide the feature C  matrix into list of features L
    all_real_features = sp.vstack((train_eval_features, test_features))
    orig_features = divide_features(all_real_features, divisions)
     
    # 3. get top n proposed features from L'
    propositions = get_top_n_popositions(pred_features) # should be AT LEAST 100 for authors 
     
    # 4. get real features from L
    real_features = get_real_propositions(orig_features)
    # k=0; [x for x in real_features[0][k] if x in propositions[0][k]]
     
    # 5. Filter relevant papers P
    intersections = get_intersections(propositions, real_features, num_features)
    
    # 6. combine intersections
    combination = combine_intersections(intersections)
    
    # 7. Sort P by cosine similarity
    cosine = get_cosine_matrix("../data/aan/paper_abstract_vec/paper_abstract_vec.features.pkl",
                               "../data/aan/paper_abstract_vec/paper_abstract_vec.train_eval_test_dict.pkl")
    best_indices = sort_combination(combination, cosine)
     
    # 8. Get recall
    get_recall(best_indices, adj_sort)
    print " "
    print "++++ RAW COSINE SIM ++++"
    best_indices2 = sort_combination(np.ones(combination.shape), cosine)
    get_recall(best_indices2, adj_sort)

    # additional stuff
    #adj_test = adj_sort[num_train_eval:, :]
    #real_ones = get_real_indices(adj_test)
    
    #score = 0.
    #for row_idx in range(len(real_ones)):
    #    y_real = set(real_ones[row_idx])
    #    y_pred = set(sp.coo_matrix(intersections[0][row_idx]).col )
    #    comm = y_real.intersection(y_pred)
    #    if len(y_real) > 0:
    #        score += (1.*len(comm)) / len(y_real)
    #        
    # 
    #print "Score: {}".format(score/len(real_ones))

# new structure
# Recall@20: 0.260566714365
# Recall@40: 0.330982779416
# Recall@60: 0.379499361533
# Recall@80: 0.413444119793
# Recall@100: 0.444483726151

# raw doc2vec
# Recall@20: 0.287773193677
# Recall@40: 0.369654924346
# Recall@60: 0.42279697226
# Recall@80: 0.461553411698
# Recall@100: 0.491106238535

    print "[INFO] Start calculating recall finished" 

    
if __name__ == '__main__':
    
    path = 'aan/paper_title/paper_title'
    try:
        path = sys.argv[1] # path
    except:
        print "No path given. Using: {}".format(path)
        
    perf(path, 'arga_ae')
    
    
