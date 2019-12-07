import numpy as np
import pickle as pkl
import random
import heapq
import operator
import sys
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

def get_top_n_indices(n, iterabl):
    final_list = zip(*heapq.nlargest(n, enumerate(iterabl), key=operator.itemgetter(1)))[0]
    return final_list
        

"""
This function calulates the similarity matrix based on the cos-sim
##### INPUT #####
:sim_mat      A numpy array of the similarities
##### RETURN #####
:sim_indices  The indicies of the most similar documents
Example: [[0.2, 0.7, 0.1], [0.5, 0.3, 0.9]] ==> [[1, 0, 2], [2, 0, 1]]    
"""
def get_most_similar_indices_list(sim):
    sim_indices = list()
    for i in range(sim.shape[0]):
        sim_indices.append(get_top_n_indices(100, sim[i]))
    return sim_indices

"""
This function calculates the citations from the adjacency matrix
##### INPUTs #####
:m       The adjacency matrix (sparse or numpy matrix)
##### RETURN #####
:citations A list of lists representing the citations.
Example: [[0, 0, 1], [1, 0, 1]] ==> [[2], [0, 2]] 
"""
def get_real_cites_list(m):
    N = m.shape[0]
    citations = list()
    for i in range(N):
        citations.append(m[i].tocoo().col)
    return citations

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
def load_data(dataset, model):
    train_eval_emb = pkl.load(open('result/{}.train_eval_embeddings.{}'.format(dataset, model), 'r')) # scipy sparse matrix
    test_emb = pkl.load(open('result/{}.test_embeddings.{}'.format(dataset, model), 'r')) # scipy sparse matrix
    orig = pkl.load(open('result/{}.adj_sort.{}'.format(dataset, model), 'r')) # scipy sparse matrix    
    
    return train_eval_emb, test_emb, orig

def output_recall(list_of_citations, sim_indices):
    """
    This function outputs the recall of the recommendations
    ##### INPUTS #####
    :list_of_citations    A list of lists containing the real citation indices
    :sim_indices          A list of lists containing the recommendation indices
    """
    # N is the number of nodes / vertices
    N = len(list_of_citations)    
    
    # For every recall we want...
    for recall in [20, 40, 60, 80, 100]:
        # tmp is the sum of the recall of every node. We use it to get the average
        tmp = 0.0
        # For every Node ...
        for i in range(N):
            # get the real citations
            real_cit = list_of_citations[i] # [?]
            # only if AT LEAST ONE citation has been made we can calculate the recall
            if len(real_cit) > 0:
                # get the predicted citations
                pred_cit = sim_indices[i][0:recall] # [N]
                # get the intersection of both: predicted and real
                cup = set(real_cit) & set(pred_cit)
                # calculate the recall and add it to tmp
                tmp += (len(cup)*1.0) / len(real_cit)
        # take the average recall by dividing tmp by N
        tmp /= N
        print("Recall@{} = {}".format(recall, tmp))

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
    
    print "MRR = {0:.5f}".format(_sum)
            
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
    
    print "MAP = {0:.5f}".format(_MAP)

def output_performance(cos_sim, sub_orig):
    print("[INFO] get list_of_citations")  
    list_of_citations = get_real_cites_list(sub_orig) # [p x ?]
    
    print("[INFO] get sim_indices")
    sim_indices =  get_most_similar_indices_list(cos_sim) # [N x N]
    
    output_MAP(list_of_citations, sim_indices)
    output_MRR(list_of_citations, sim_indices)
    output_recall(list_of_citations, sim_indices)
    
    
"""
This function will calculate the performance of the citation.
For every paper it uses the cos-sim of the embeddings to find similar papers and than prints the recall.
##### INPUTS ####
:dataset        The name of the dataset. e.g 'cora'
:model          The name of the model e.g. 'arga_ae'
:train_eval_emb The train and eval embeddings
:test_emb       The test embeddings
:orig           The original adj. matrix
"""
def perf(dataset, model, train_eval_emb=None, test_emb=None, orig=None):
    print "[INFO] Start calculating recall ..." 
    
    # if the matrices are not provided ==> load them
    if train_eval_emb==None or test_emb==None or orig==None:
        train_eval_emb, test_emb, orig = load_data(dataset, model)
 
    # convert emb to a numpy matrix
    if not type(train_eval_emb) == np.ndarray:
        train_eval_emb = train_eval_emb.toarray()
    if not type(test_emb) == np.ndarray:
        test_emb = test_emb.toarray()
  
    # create the similarities between the test embeddings and the train-eval embeddings
    sim = cosine_similarity(test_emb, train_eval_emb)
    
    # take the X like shown here:
    # 0 0 0 0 0
    # 0 0 0 0 0
    # 0 0 0 0 0
    # X X X 0 0
    # X X X 0 0
    sub_orig = orig[train_eval_emb.shape[0]:, :train_eval_emb.shape[0]]
    
    output_performance(sim, sub_orig)

    
if __name__ == '__main__':
    
    path = "aan/paper_title/paper_title"
    try:
        path = sys.argv[1] # path
    except:
        print "No path given. Using: {}".format(path)
        
    perf(path, 'arga_ae')
    
    
