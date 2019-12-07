#!/usr/bin/env python
# coding=utf-8

import cPickle as pkl
import networkx as nx
import scipy.sparse as sp
import numpy as np
import sys
import os
import re
import io
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#for the doc2vec
from gensim.models.doc2vec import Doc2Vec
from sklearn.feature_extraction import stop_words


class Paper:
    def __init__(self, pap_id, authors, title, venues, year):
        self.id = pap_id
        self.authors = authors
        self.title = title 
        self.venues = venues 
        self.year = year 
    
    def __str__(self):
        return "{}: {} {} {} {}".format(self.id, str(self.authors), self.title, self.venues, self.year)

"""
This checks, the path to the file exists.
##### INPUTS #####
:file_name     The name of the file to be checked
##### RETURN #####
:bool          Returns true, iff the path exists.
"""
def path_to_file_exists(file_name):
    path_arr = file_name.split("/")
    path = "/".join(path_arr[0:-1])
    return os.path.exists(path)
    
"""
Creates the path to the file
"""
def create_path_to_file(file_name):
    path_arr = file_name.split("/")
    path = "/".join(path_arr[0:-1])
    return os.mkdir(path)

def preprocess_text(text, stop_words):
        text = re.sub("\r\n", "", text).lower()
        text = re.sub("[\[\].,*\'()\{\};:!?]", "", text).lower()  
        text = re.sub("-", " ", text)
        text = text.split()
        text = [w for w in text if (not w in stop_words) and len(w) > 3]
        text = " ".join(text)
        return text

"""
This function takes the metadata from 'acl-metadata' and retrieves all the data.
It stores it in a list of Paper and dumps is via pickle.
It also creates a list of all paper_ids and stores it in the result_ids_name file
"""
def process_metadata(filename, resultname):
    print "[INFO] Process Metadata..."
    f = open(filename, 'r')
    papers = dict()   
    pap_id = None; authors = []; title = None; venues = []; year = None
    for line in f:
        if line == '\n':
            paper = Paper(pap_id, authors, title, venues, year)
            if not papers.has_key(paper.id):
                papers[paper.id] = paper
            else:
                papers[paper.id].venues.extend(venues)
            pap_id = None; authors = []; title = None; venues = []; year = None
        else:
            pap_type, _ , data = line.split(' ', 2)
            if "id" in pap_type:
                pap_id = data[1:-2]
            elif "author" in pap_type:
                raw_auth = data[1:-2]
                split_auth = raw_auth.split('; ')
                for author in split_auth:
                    authors.append(author.replace(" ", "")) #remove all whitespaces
            elif "title" in pap_type:
                title = data[1:-2]
            elif "venue" in pap_type:
                venues.append(data[1:-2])
            elif "year" in pap_type:
                year = int(data[1:-2])
                
    pkl.dump(papers.values(), open(resultname, 'w'))
    f.close()
    print "[INFO] Process Metadata finished"

"""
In the result of the process_metadata(...) the authors dont have any whitespaces.
Therefore we have to remove them also in the author_ids.txt file.
"""
def process_author_ids(filename, resultname):
    print "[INFO] Process Author IDs..."
    f_in = open(filename, 'r')
    f_out = open(resultname, 'w')
    
    for line in f_in:
        n, auth = line.split("\t", 1)
        new_line = "{}\t{}".format(n, auth.replace(" ", ""))
        f_out.write(new_line)
    print "[INFO] Process Author IDs finished"
    
"""
This function creates a list of all relevant papers.
A paper can only be relevant iff:
(1) Title, Author and Venue is given
(2) The paper has more then 2 citations
##### INPUTS #####
:processed_acl_meta         The name of the file of the processed meta data (pickle format)
:processed_author_ids       The name of the file of the processed author ids
:papers_text                The name of the directory of all the abstracts
:min_year                   The minimum year of publishing of the paper
:max_year                   The maximum year of publishing of the paper
:relevant_paper_ids         The name of the file where the the relevant IDs will be stored
:min_citations              Only the papers with >=min_citation citations will be considered relevant
##### OUTPUT #####
"""
def create_relevant_paper_id_list(processed_acl_meta, processed_author_ids, papers_text, acl_list, min_year, max_year, relevant_paper_ids, min_citations=2):
    print "[INFO] Creating relevant paper id list ..."
    papers = pkl.load(open(processed_acl_meta, 'r'))
    file_author_ids = open(processed_author_ids, 'r')
    file_relevant_papers = open(relevant_paper_ids, 'w')
    file_acl_list = open(acl_list, 'r')
        
    num_rel_pap = 0
    
    """
    Check if a file has >=n words.
    """
    def file_has_n_words(file_name, n):
        with open(file_name, 'r') as f:
            count = 0
            for l in f:
                count += len(l.split())
                if count >= n:
                    return True 
            return False
    
    """
    get the number of citations for a paper
    """
    def get_num_citations(acl_file):
        num_cit = dict()
        for l in acl_file:
            p1, _, _ = l.split()
            if num_cit.has_key(p1):
                num_cit[p1] += 1
            else:
                num_cit[p1] = 1
        return num_cit
        
    # define a lookup dictionary for the author ids
    
    num_cit = get_num_citations(file_acl_list)
    
    author_ids = dict()
    for line in file_author_ids:
        n, auth = line.strip().split("\t", 1)
        author_ids[auth] = n
        
    for paper in papers:
        
        # check venue, id, title and year
        if paper.id == None or len(paper.venues) == 0 or paper.title == None or paper.year == None or len(paper.authors) == 0:
            continue
        
        # check num citations:
        if (not num_cit.has_key(paper.id)) or (num_cit[paper.id] < min_citations):
            continue
        
        # check year
        year = paper.year
        if year < min_year or year > max_year:
            continue    
        
        # if any of the authors is not contained ==> drop paper
        not_contained = False # argh.. this is ugly
        for a in paper.authors:
            if not author_ids.has_key(a):
                not_contained = True
                break
        if not_contained:
            continue
                
        # check abstract size
        try:
            file_name = "{}/{}.txt".format(papers_text, paper.id)
            stat = os.stat(file_name)
            if stat.st_size > 0: 
#             if file_has_n_words(file_name, 0):
                file_relevant_papers.write("{}\n".format(paper.id))
                num_rel_pap += 1
        except:
            pass
        
    file_author_ids.close()
    file_relevant_papers.close()
    file_acl_list.close()
    print "[INFO] Creating relevant paper id list finished. {} papers are relevant of {}".format(num_rel_pap, len(papers))

"""
This function decides based on the metadata if a paper is for training or testing.
The output file contains a dictionary: <index> ==> <'train'/'test'/'eval'>
##### INPUTS #####
:acl_metadata               The name of the file which contains the metadata
:paper_indices_name         The name of the file which contains the index of a paper in the adj. matrix
:test_year                  All papers with paper.year >= test_year will be labeled as 'test'
:eval_year                  All papers with test_year > paper.year >= eval_year will be labeled as 'eval'
:result_name                The name of the file where the result will be stored.
"""
def create_train_val_test_dict(acl_metadata, paper_indices_name, test_year, eval_year, result_name):
    print "[INFO] Creating train test dict ..."
    papers = pkl.load(open(acl_metadata))
    paper_indices = pkl.load(open(paper_indices_name))
    
    train_test_dict = dict()
    
    train = 0
    evalu = 0
    test = 0
    i = 0
    for paper in papers:
        # A paper my be contained in the metadata but not in the paper_indices, since the paper_indices
        # were filtered for relevant papers
        try:
            idx = paper_indices[paper.id]
            if type(paper.year) != int:
                print "[ERROR] paper.year has wrong type!"
                return
            if paper.year < eval_year:
                train_test_dict[idx] = 'train'
                train += 1
            elif paper.year < test_year:
                train_test_dict[idx] = 'eval'
                evalu += 1
            else:
                train_test_dict[idx] = 'test'
                test += 1
            i += 1
        except:
            pass
            
    pkl.dump(train_test_dict, open(result_name, 'w'))
    print "[INFO] Creating train test dict finished. Partition {}(Train)<==>{}(Eval)<==>{}(Test)".format(train, evalu, test)

"""
This function creates the network for the Paper ==> Author relation.
It takes as input the PROCESSED author_ids from process_author_ids(...) and
the acl-metadata as pickle format from process_metadata(...)
"""
def create_paper_author_network(author_ids_file, acl_metadata_pickle, resultname):
    print "[INFO] Create Paper Author Network..."
    aif = open(author_ids_file, 'r')

    # First we create a dictionary for the authors: 'Name' ==> 'Num-ID'
    author_dict = dict()
    for line in aif:
        n, auth = line.split("\t", 1)
        auth = auth.strip()
        author_dict[auth] = n
    aif.close()
        
    # Now we load the papers from the pickle file. This results in a list of Paper
    papers = pkl.load(open(acl_metadata_pickle, 'r'))
    
    output_file = open(resultname, 'w')
    
    # For every paper get the make a entry 'E01-2345 ==> <auth-id>' in the result file
    not_found = 0
    found = 0
    for paper in papers:
        id_pap = paper.id
        for auth in paper.authors:
            try:
                id_auth = author_dict[auth]
                output_file.write("{} ==> {}\n".format(id_pap, id_auth))
                found += 1
            except:
                # print("Author not found: ", auth)
                not_found += 1
    print"[INFO] Total authors not found: ", not_found # 915
    print"[INFO] Total authors found: ", found # 61702
            
    output_file.close()
    print "[INFO] Create Paper Author Network finished"


"""
This function creates the network for the Paper ==> Venue relation.
It takes as input the acl-metadata as pickle format from process_metadata(...)
"""   
def create_paper_venue_network(acl_metadata_pickle, resultname):
    print "[INFO] Create Paper Venue Network..."
    # First we load the papers from the pickle file. This results in a list of Paper
    papers = pkl.load(open(acl_metadata_pickle, 'r'))    
    output_file = open(resultname, 'w')
    
    # For every paper get the make a entry 'E01-2345 ==> <auth-id>' in the result file
    for paper in papers:
        output_file.write("{} ==> {}\n".format(paper.id, paper.venues))            
    output_file.close()
    print "[INFO] Create Paper Venue Network finished"

"""
This function takes as input the acl.txt file and stores a networkx graph in the resultfile
##### INPUTS #####
:acl_file                  The processed acl file
:relevant_paper_ids        The list of the relevant papers
:paper_indices_result_file The name of the file where the indices of the nodes will be stored.
:adjacency_matrix          The name of the file where the adjacency matrix will be stored
"""
def create_paper_graph(acl_file, relevant_paper_ids, paper_indices_result_file, adjacency_matrix):
    print "[INFO] Create Paper Graph..."
    acl = open(acl_file, 'r')
    file_rel_paper_ids = open(relevant_paper_ids, 'r')
    
    # first we need to know, how many nodes there are:
    paper_ids = dict()
    for l in file_rel_paper_ids:
        paper_ids[l.strip()] = len(paper_ids)
        
    # create a empy Graph  
    G = nx.Graph()
#     G = nx.DiGraph()
    
    # add all the nodes
    for node in paper_ids.keys():
        G.add_node(node)
    
    # now for the links:    
    acl.seek(0) # reset the cursor
    for l in acl:
        p1, _, p2 = l.split()
        if paper_ids.has_key(p1) and paper_ids.has_key(p2):
            G.add_edge(p1, p2)
    
    # dump the indices
    paper_indices = dict()
    nodes = list(G.nodes())
    for k in paper_ids.keys():
        paper_indices[k] = nodes.index(k)
    pkl.dump(paper_indices, open(paper_indices_result_file, 'w'))
    
    # dump the adjacency matrix
    A = nx.adjacency_matrix(G)
    pkl.dump(A, open(adjacency_matrix, 'w'))
    
    file_rel_paper_ids.close()
    acl.close()
    print "[INFO] Stored adjacency matrix of size [{}x{}]".format(len(G.nodes()), len(G.nodes()))
    print "[INFO] Create Paper Graph finished"
     
"""
This function creates
a feature-matrix: Node <==> Author and 
a adjacency-matrix: Node <==> Node.
It dumps them via pickle as a sparse matrix
##### INPUTS #####
:paper_indices               The name of the file with the indices of the papers in the adj. matrix
:paper_author_network_name   The name of the paper_author_nework: E.g. "processed/paper_author_network.txt"
:result_features_name        The result name: E.g. "processed/aan_paper_author.features.pkl"
"""
def dump_paper_author(paper_indices, paper_author_network_name, result_features_name, max_author=0):
    print "[INFO] Dump Paper Author Features..."
    #First we have to define the order of the nodes in the matrix.
    # We therefore use the Graph.nodes() function:
    paper_indices = pkl.load(open(paper_indices, 'r')) # dict: "E98-1001" ==> 123
    
    # pan is the Paper-Author-Network
    pan = open(paper_author_network_name, 'r')
    
    # get all the authors, and count their frequency
    authors = dict()   
    for l in pan:
        # every line has the format: "E01-1234 ==> 1234"
        _, _, auth = l.split()
        if auth in authors:
            authors[auth] = authors[auth]+1
        else:
            authors[auth] = 0
    
    # get the 500 most frequent authors
    sort = sorted(authors, key=lambda x: authors[x], reverse=True)
    if max_author != 0:
        sort = sort[0:max_author]
    
    # get the number of authors, which have at least 2 papers written
    idx = 0
    for s in sort:
        if authors[s] > 1:
            idx += 1
        else:
            break
    
    authors = sort[0 : min(idx, len(sort))]
    
    #define fixed column-index for every author:
    author_indices = dict()
    for i, auth in enumerate(authors):
        author_indices[auth]= i
    
    # create the feature_matrix
    feature_matrix = sp.lil_matrix((len(paper_indices.keys()), len(author_indices.keys())), dtype=np.float32)
    # add the links between paper and author:
    pan.seek(0)
    for l in pan:
        paper, _, auth = l.split()
        # about 3766 papers are in the acl-metadata. but are not contained in the acl list.
        try:
            feature_matrix[paper_indices[paper], author_indices[auth]] = 1.0
        except:
            pass
        
    pkl.dump(feature_matrix.tocsr(), open(result_features_name, 'w'))
    pan.close()
    print "[INFO] Dumped Feature Matrix PAPER <==> AUTHOR  of size {}".format(feature_matrix.shape)
    print "[INFO] Dump Paper Author Features finished"
    

"""
This function creates
a feature-matrix: Node <==> Venue and 
a adjacency-matrix: Node <==> Node.
It dumps them via pickle as a sparse matrix
##### INPUTS #####
:max_authors                 The maximum number of authors to take.
:paper_indices               The name of the file with the indices of the papers in the adj. matrix
:paper_venue_network_name    The name of the paper_author_nework: E.g. "processed/paper_author_network.txt"
:result_features_name        The result name: E.g. "processed/aan_paper_author.features.pkl"
"""
def dump_paper_venue(max_venues, paper_indices, paper_venue_network_name, result_features_name):
    print "[INFO] Dump Paper Venue Features..."
    # First we have to define the order of the nodes in the matrix.
    # We therefore use the Graph.nodes() function:
    paper_indices = pkl.load(open(paper_indices, 'r'))
    
    # pan is the Paper-Author-Network
    pvn = open(paper_venue_network_name, 'r')
    
    # get all the venues, and count their frequency
    venues = dict()   
    for l in pvn:
        # every line has the format: "E01-1234 ==> 1234"
        _, _, venue = l.split(" ", 2)
        if venue in venues:
            venues[venue] = venues[venue]+1
        else:
            venues[venue] = 0
    
    # get the max_venues most frequent venues
    sort = sorted(venues, key=lambda x: venues[x], reverse=True)
    venues = sort[0 : min(max_venues, len(sort))]
    
    #define fixed column-index for every author:
    venue_indices = dict()
    for i, venue in enumerate(venues):
        venue_indices[venue]= i
    
    # create the feature_matrix
    feature_matrix = sp.lil_matrix((len(paper_indices.keys()), len(venue_indices.keys())), dtype=np.float32)
    # add the links between paper and author:
    pvn.seek(0)
    unfound_venues = set()
    for l in pvn:
        paper, _, venue = l.split(" ", 2)
        # about 3766 papers are in the acl-metadata. but are not contained in the acl list.
        try:
            feature_matrix[paper_indices[paper], venue_indices[venue]] = 1.0
        except:
            unfound_venues.add(paper)
            pass
    
    # Note: Papers can be published in multiple venues!
    pkl.dump(feature_matrix.tocsr(), open(result_features_name, 'w'))
    pvn.close()
    print "[INFO] Dumped Feature Matrix PAPER <==> VENUE  of size {}".format(feature_matrix.shape)    
    print "[INFO] Dump Paper Venue Features finished"
           
"""
This function takes the pre-processed BOW-list and the Nodes.
It calculates the adjacency matrix and the feature Matrix: Node <==> Word
##### INPUTS #####
:paper_indices               The name of the file with the indices of the papers in the adj. matrix
:bow_list_name               The name of the bow list: E.g. "processed/bow_list.txt"
:directory_name              Name of the directory of all files. E.g. "../aan/papers_text"
:result_features_name        The result name: E.g. "processed/aan_paper_bow.features.pkl"
"""
def dump_paper_bow(max_words, paper_indices, stop_words, directory_name, result_features_name):
    print "[INFO] Dump Paper Bow Features..."
    paper_indices = pkl.load(open(paper_indices, 'r'))  
    stop_words = pkl.load(open(stop_words, 'r'))
    
    class Corpus:
        def __init__(self, paper_indices, stop_words, directory_name):
            self.paper_indices = paper_indices
            self.stop_words = stop_words
            self.directory_name = directory_name
            
        def __iter__(self):
            # create an inverse dict of paper indices:
            sorted_ids = sorted(paper_indices.keys(), key=lambda paper_id: paper_indices[paper_id])
            for paper_id in sorted_ids:
                path = "{}/{}.txt".format(self.directory_name, paper_id)
                abstract = ""
                with open(path, 'r') as f:
                    abstract = f.read()
                yield preprocess_text(abstract, self.stop_words)
         
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=max_words, min_df=1, max_df=1.)
    vectors = tfidf_vectorizer.fit_transform(Corpus(paper_indices, stop_words, directory_name))
                
    pkl.dump(vectors, open(result_features_name, 'w'))
    
    print "[INFO] Dump Paper Bow Features finished"


def feature_to_label(feature_name, adj_matrix_name, result_name, ceil=False, clip=0.):
    """
    This function creates labeled data.
    For a row R, which represents a paper P in the feature matrix C, the labeled data
    will inherit all features from the citations made in P and add it to the labeled data.
    E.G.
    C =        Citations=
    [0, 1, 0]  0 ==> 1
    [0, 0, 1]
    result:
    [0, 0, 1]
    [0, 0, 0]
    ##### INPUTS #####
    :features                The name of the file of the features
    :adj_matrix_name         The name of the file of the adj matrix
    :result                  The name of the file where the results will be dumped
    :ceil=False              If set to true, the feature label will be ceiled just after it has been loaded.
    :clip=0.                 It set to a value x!=0.0 the result will be cliped to x before it is dumped.
    """
    print "[INFO] Feature to label ({}) ...".format(feature_name)
    # create the directories if necessary
    assert(os.path.isfile(feature_name))
    assert(os.path.isfile(adj_matrix_name))
    if not path_to_file_exists(result_name):
        create_path_to_file(result_name)
        
    # load the other structures
    with open(feature_name) as feature_file:
        features = pkl.load(feature_file)
        if ceil:
            features = np.ceil(features)
    with open(adj_matrix_name) as adj_matrix_file:
        adj_matrix = pkl.load(adj_matrix_file)
    
    # emtpy result matrix    
    result = np.matmul(adj_matrix.toarray(), features.toarray())
    
    if clip != 0.:
        result = np.clip(result.toarray(), 0, clip)
        
    result = sp.csr_matrix(result)
    pkl.dump(result, open(result_name, 'w'))
    
    print "[INFO] Feature to label ({}) finished".format(feature_name)

"""
This function creates
a feature-matrix: Node <==> BOW(title) and 
a adjacency-matrix: Node <==> Node.
It dumps them via pickle as a sparse matrix
##### INPUTS #####
:max_words                   The maximum number of words to take for the BOW.
:paper_indices               The name of the file with the indices of the papers in the adj. matrix
:acl_meta                    The name of the file of the  acl metadata in the pickle format
:stop_words                  The name of the file of the stop words
:result_features_name        The result name: E.g. "processed/aan_paper_author.features.pkl"
"""
def dump_paper_title(max_words, paper_indices_name, acl_meta, stop_words, result_feature_name):
    print "[INFO] Dump paper title ..."
    paper_indices = pkl.load(open(paper_indices_name, 'r'))    
    stop_words = pkl.load(open(stop_words, 'r'))    
    meta = pkl.load(open(acl_meta, 'r'))
    
    corpus = [None] * len(paper_indices)
    for paper in meta:
        try:
            idx = paper_indices[paper.id]
            corpus[idx] = preprocess_text(paper.title, stop_words)
        except:
            pass
                
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=max_words, min_df=1, max_df=1.)
    vectors = tfidf_vectorizer.fit_transform(corpus)
                
    pkl.dump(vectors, open(result_feature_name, 'w'))
    print "[INFO] Dump paper title finished"

def dump_paper_year(acl_meta_name, paper_indices_name, result_name):
    """
    This function dumps the year feature matrix
    ##### INPUTS #####
    :acl_mat_name           The name of the file of the acl-metadata
    :paper_indices_name     The name of the file of the paper_indices
    :result_name            The name of the file of the result
    """
    
    print "[INFO] Dumping year feature matrix ..."
    
    # Load the data
    with open(acl_meta_name) as acl_meta_file:
        acl_meta = pkl.load(acl_meta_file)
    with open(paper_indices_name) as paper_indices_file:
        paper_indices = pkl.load(paper_indices_file)
        
    # create a matrix
    year_feature_matrix = sp.lil_matrix(( len(paper_indices), 1 ))
    
    # Iterate through all the papers and add the year to the matrix if they are relevant
    for paper in acl_meta:
        paper_id = paper.id
        try:
            paper_index = paper_indices[paper_id]
            year_feature_matrix[paper_index, 0] = (paper.year - 1965) / (2020.0-1965.0)
        except:
            pass
    
    # Convert from lil to CSR
    year_feature_matrix = sp.csr_matrix(year_feature_matrix)
    
    # Dump the result
    with open(result_name, 'w') as result_file:
        pkl.dump(year_feature_matrix, result_file)
        
    print "[INFO] Dumping year feature matrix finished"
    

"""
This function creates the vectors for the documents (Doc2Vec)
##### INPUTS #####
:directory_name              The name of the papers e.g. "raw/papers_text"
:paper_indices_name          The name of the file with the indices of the papers in the adj. matrix
:stop_words                  The name of the file of the stop words
:result_name                 The name of the file where the vectors should be stored.
"""
def title2vec(acl_metadata, paper_indices_name, file_stop_words, result_name):
    print "[INFO] Title 2 Vec..."
    
    stop_words = pkl.load(open(file_stop_words, 'r'))    
    paper_indices = pkl.load(open(paper_indices_name, 'r'))
    acl_meta = pkl.load(open(acl_metadata, 'r'))
     
    MAX = len(acl_meta)
    
    model = Doc2Vec.load("apnews_dbow.bin")
    embedding = sp.lil_matrix((len(paper_indices), model.vector_size))
    count = 0
    
    for paper in acl_meta:
        try:
            text = preprocess_text(paper.title, stop_words)
            embedding[paper_indices[paper.id], :] = model.infer_vector(text.split())
            count += 1
        except:
            pass
    
    embedding = normalize(embedding, norm='l1', axis=1)
    embedding = embedding.tocsr()
    
    pkl.dump(embedding, open(result_name, 'w'))
    print "[INFO] Title 2 Vec finished. {}/{} vecs were inserted into the embedding {}".format(count, MAX, embedding.shape)

"""
This function creates the vectors for the documents (Doc2Vec)
##### INPUTS #####
:directory_name              The name of the papers e.g. "raw/papers_text"
:paper_indices_name          The name of the file with the indices of the papers in the adj. matrix
:stop_words                  The name of the file of the stop words
:result_name                 The name of the file where the vectors should be stored.
"""
def abstract2vec(directory_name, paper_indices_name, file_stop_words, result_name):
    print "[INFO] Papers 2 Vec..."
    
    stop_words = pkl.load(open(file_stop_words, 'r'))
    paper_indices = pkl.load(open(paper_indices_name, 'r'))
    nodes = paper_indices.keys() # = ['E01-1001', 'E01-1002', ...]
    
    MAX = len(nodes)
    
    class MyCorpus():
        
        def __init__(self, dir_name, stop_words):
            self.dir_name = dir_name
            self.stop_words = stop_words
            
        def __iter__(self):
            i = 0
            for paper_id in nodes[0:MAX]:
                path = "{}/{}.txt".format(self.dir_name, paper_id)
                try:
                    # open file and get abstract
                    f = io.open(path, 'r')
                    abstract = f.read()
                    f.close()
                    
                    # get a TaggedDocument based on the abstract
                    abstract = preprocess_text(abstract, self.stop_words)
                    
                    # yield the TaggedDocument
                    yield (paper_id, abstract)
                except:
                    print("could not open {}".format(path))
                    pass
                i = i+1
                if i % 100 == 0:
                    print "[INFO] {} of {} papers".format(i, len(nodes))

#     tagged_data = MyCorpus(directory_name)
#     
#     # TODO: do this with the pretrained Doc2Vec model
#     
#     max_epochs = 5
#     vec_size = 150
#     
#     model = Doc2Vec(dm = 1, min_count=5, window=10, size=vec_size, sample=1e-4, negative=10)
#       
#     model.build_vocab(tagged_data)
#     
#     for _ in range(max_epochs):
#         model.train(tagged_data, epochs = 10, total_examples=model.corpus_count)
#         # decrease the learning rate
# #         model.alpha -= 0.0002
#         # fix the learning rate, no decay
# #         model.min_alpha = model.alpha        print('Training iteration {0}'.format(epoch))
# 
#     model.save("abstract2vec-model.bin")
#     print("[INFO] Model Saved")
#     
#     # get the embeddings
#     embedding = sp.lil_matrix((len(nodes), vec_size))
#     for i, n in enumerate(nodes[0:MAX]):
#         embedding[paper_indices[n], :] = model.docvecs[i]

    # load pre-trained model
    model = Doc2Vec.load("apnews_dbow.bin")
    embedding = sp.lil_matrix((len(paper_indices), model.vector_size))
    
    count = 0
    
    for paper_id, abstract in MyCorpus(directory_name, stop_words):
        try:
            embedding[paper_indices[paper_id]] = model.infer_vector(abstract.split())
            count += 1
        except:
            pass
    
    embedding = normalize(embedding, norm='l1', axis=1)
    embedding = embedding.tocsr()
    
    pkl.dump(embedding, open(result_name, 'w'))
    print "[INFO] Papers 2 Vec finished"

"""
Calculates Cos-sim for big matrices
"""
def cosine_similarity_n_space(m1, m2, batch_size=100):
    assert m1.shape[1] == m2.shape[1]
    m1 = m1.astype(dtype=np.float32)
    m2 = m2.astype(dtype=np.float32)
    ret = sp.lil_matrix((m1.shape[0], m2.shape[0]), dtype=np.float32)
    for row_i in range(0, int(m1.shape[0] / batch_size) + 1):
        start = row_i * batch_size
        end = min([(row_i + 1) * batch_size, m1.shape[0]])
        if end <= start:
            break # cause I'm too lazy to elegantly handle edge cases
        rows = m1[start: end]
        sim = cosine_similarity(rows, m2) # rows is O(1) size
        sim = sp.csr_matrix(np.round(sim, 1))
        ret[start: end] = sim
        print "[INFO] Row {} of {}".format(row_i, int(m1.shape[0] / batch_size) + 1)
    return ret.tocsr()
 
def dump_author_authors(name_adj_matrix, name_paper_author_feat, name_result_file):
    """
    This function dumps a feature matrix where each paper has as attribute the cited authors in the past
    from the authors writing the paper.
    ##### INPUTS #####
    :name_author_network          The name of the file of the processed author ==> author relation
    :name_paper_author_feat       The name of the file of the paper_author feature matrix
    :name_paper_indices           The name of the file of the paper indices
    :name_author_indices          The name of the file of the author indices
    :name_result_file             The name of the result file
    """
    
    print "[INFO] Dump auth auth ..."
    
    # first we load all the files
    with open(name_adj_matrix) as file_adj_matrix:
        adj_matrix = pkl.load(file_adj_matrix)
    with open(name_paper_author_feat) as file_paper_author_feat:
        paper_author_feat = pkl.load(file_paper_author_feat)
        
    # create paper ==cited==> authors relation matrix
    pap_cited_auth = np.matmul(adj_matrix.toarray(), paper_author_feat.toarray())
    del adj_matrix
    
    # create author ==cited==>author relation matrix
    author_cited_author = np.matmul(np.transpose(paper_author_feat.toarray()), pap_cited_auth)
    del pap_cited_auth
    
    # create papers authors ==cited==> authors relation matrix
    auth_auth_feat = np.matmul(paper_author_feat.toarray(), author_cited_author)
    del paper_author_feat
    del author_cited_author    
    
    auth_auth_feat = sp.csr_matrix(auth_auth_feat)
    with open(name_result_file, 'w') as result_file:
        pkl.dump(auth_auth_feat, result_file)
    
    print "[INFO] Dump auth auth finished"
    
"""
This function takes the doc-vectors and calculates a cos-sim square matrix from it.
##### INPUTS #####
:docvec_name         The name of the document where the vectors are stored.
:result_name         The name of the file where the vectors should be stored.
"""
def doc_vec_to_adj(docvec_name, result_name):
    print "[INFO] Doc to vec adj..."
    vecs = pkl.load(open(docvec_name, 'r'))
    vecs = vecs.astype(dtype=np.float16)
    similarities = cosine_similarity_n_space(vecs, vecs)
    f = open(result_name, 'w')
    np.save(f, similarities)
    f.close()
    # can be loaded by NOTE: Very memory consuming: +3GB
    # f = open(result_name, 'r')
    # sim = np.load(f)
    # f.close()
    print "[INFO] Doc to vec adj FINISEHD"
    
"""
This function joins to feature Matrixec of the size [N x p] and [N x q] to [N x (p+q)]
##### INPUTS #####
:feature_max_res     The name of the resulting feature matrix
:feature_matrices    The name of the first feature matrix
"""
def join_feature_matrices(feature_mat_res, divisions_name, *feature_matrices):
    print "[INFO] Joining features:"
    assert(len(feature_matrices) > 0)
    
    res = pkl.load(open(feature_matrices[0]))
    divisions = [res.shape[1]]
    
    for i in range(1, len(feature_matrices)):
        with open(feature_matrices[i]) as f:
            to_add = pkl.load(f)
            res = sp.hstack((res, to_add)).tocsr()
            divisions.append(to_add.shape[1])
        
    pkl.dump(res, open(feature_mat_res, 'w'))
    pkl.dump(divisions, open(divisions_name, 'w'))
    print "[INFO] Joining features finished. Final size: {}".format(res.shape)
    
def dump_history(result_name, paper_indices_name, acl_metadata_name, adj_matrix_name, paper_authors_name, history):
    """
    This function creates a history of citations for the authors:
    ##### INPUTS #####
    :restult_name          The name of the result file
    :history               A list of integers defining the time intervals for the history. E.g. [1980, 1990, 2000, 2005, 2011]
    ##### OUTPUT #####
    :history               A list of sparse csr. matrices where each matrix is the citation history for a time interval.
    """
    
    print "[INFO] Dumping history: {} ...".format(history)
    
    with open(adj_matrix_name) as adj_file:
        adj_matrix = pkl.load(adj_file)
        
    with open(paper_authors_name) as authors_file:
        paper_authors = pkl.load(authors_file)
        
    with open(paper_indices_name) as paper_indices_file:
        paper_indices = pkl.load(paper_indices_file)
        
    with open(acl_metadata_name) as acl_meatadata_file:
        acl_metadata = pkl.load(acl_meatadata_file)
        
    num_papers = adj_matrix.shape[0] # N
    # num_intervals = len(history)
        
    # The adj. matrix gives the relation between a paper and its citations
    # First we create (from the original adj matrix A) multiple adj matrices: [A1, A2, ...]. Each indicates the citations of a time interval.
    # Note: A == A1 + A2 + ...
    
    adj_matrices = []
    
    # fill the list with empty lil matrices
    for _ in history:
        adj_matrices.append(sp.lil_matrix((num_papers, num_papers)))
    
    for paper in acl_metadata:
        try:
            # get the Index of the paper in the adj. matrix and the year of publishion of the paper
            idx = paper_indices[paper.id]
            year = paper.year
        except:
            # if the paper is not in the relevant paper list ==>
            # it will throw an error and continue the search for important papers
            continue
        
        for i, next_interval in enumerate(history):
            # get the corresponding history adj matrix ...
            if year < next_interval:
                # and copy the row into it.
                adj_matrices[i][idx] = adj_matrix[idx]
                # now we can end the loop
                break
            # else: do nothing
    
    # Now we calculate what other citations the authors made ... [Nxm]*[NxM]' => [N x N]
    # It is necessary to set the diagonal to zero, otherwise the resulting history matrix
    # would also contain the final citations themselves
    author_citations = paper_authors * paper_authors.transpose()
    _zeros = [0.0] * num_papers
    author_citations.setdiag(_zeros)
    
    # ... and combine this with the adj. matrices
    paper_history_citations = []
    for adj_matrix in adj_matrices:
        paper_history_citation = author_citations * adj_matrix # matrix mult. not elementwise
        paper_history_citations.append(paper_history_citation.tocsr().astype(dtype=np.float32))
    
    with open(result_name, 'w') as result_file:
        pkl.dump(paper_history_citations, result_file)
    
    print "[INFO] Dumping history finished.".format(history)
    
"""
This function deletes all given files.
"""
def delete_files(iterabl):
    for f in iterabl:
        try:
            os.remove(f)
        except:
            pass
    
if __name__ == "__main__":
    
    
    dataset = "AAN"
    try:
        dataset = sys.argv[1] # path
    except:
        print "[WARNING] No dataset given. Using: {}".format(dataset)
    print "[INFO] Using: {}".format(dataset)
    
    if dataset == "DBLP":
        min_year = 1965
        max_year = 2011
        eval_year = 2008
        test_year = 2009
    elif dataset == "AAN":
        min_year = 1965
        max_year = 2013
        test_year = 2013
        eval_year = 2012
    else:
        min_year = 1965
        max_year = 2013
        test_year = 2013 # [test_year, max_year] is for testing
        eval_year = 2012 # [min_year, eval_year) is for train, [eval_year, test_year) is for eval
    
    # raw
    raw_acl = "{}/raw/acl.txt".format(dataset)
    raw_acl_meta = "{}/raw/acl-metadata.txt".format(dataset)
    raw_author_ids = "{}/raw/author_ids.txt".format(dataset)
    papers_text = "{}/raw/papers_text".format(dataset)
    stop_words = "{}/raw/stop_words.pkl".format(dataset)
    
    # processed
    processed_author_ids = "{}/processed/author_ids.txt".format(dataset)
    processed_acl_meta = "{}/processed/acl-metadata.pkl".format(dataset)
    relevant_paper_ids = "{}/processed/paper_ids.txt".format(dataset)
    processed_paper_venue_network = "{}/processed/paper_venue_network.txt".format(dataset)
    processed_paper_author_network = "{}/processed/paper_author_network.txt".format(dataset)
    processed_bow_count = "{}/processed/bow.pkl".format(dataset)
    processed_abstract_vecs = "{}/processed/paper_abstract_vec/paper_abstract_vec.features.pkl".format(dataset)
    processed_title_vecs = "{}/processed/paper_title_vec/paper_title_vec.features.pkl".format(dataset)
    processed_paper_indices = "{}/processed/paper_indices.pkl".format(dataset)
    processed_train_eval_test_dict = "{}/processed/train_eval_test_dict.pkl".format(dataset)
    adjacency_matrix = "{}/processed/adj_matrix.pkl".format(dataset)
    history_matrices = "{}/processed/history.pkl".format(dataset)
    
    # final
    fin_pap_ven_feat = "{}/processed/paper_venue/paper_venue.features.pkl".format(dataset)
    fin_pap_auth_feat = "{}/processed/paper_author/paper_author.features.pkl".format(dataset)
    fin_pap_bow_feat = "{}/processed/paper_bow/paper_bow.features.pkl".format(dataset)
    fin_pap_title_feat = "{}/processed/paper_title/paper_title.features.pkl".format(dataset)
    fin_pap_authAuth_feat = "{}/processed/paper_author_authors/paper_author_authors.features.pkl".format(dataset)
    fin_pap_year_feat = "{}/processed/paper_year/paper_year.features.pkl".format(dataset)
    fin_pap_joined_feat = "{}/processed/joined/joined.features.pkl".format(dataset)
    fin_pap_joined_divi = "{}/processed/joined/joined.divisions.pkl".format(dataset)
    fin_cossin_adj = "{}/processed/cos_sim_adj.np".format(dataset)
    
    # label
    label_pap_auth = "{}/processed/paper_author/paper_author.label.pkl".format(dataset)
    label_pap_ven = "{}/processed/paper_venue/paper_venue.label.pkl".format(dataset)
    label_pap_title = "{}/processed/paper_title/paper_title.label.pkl".format(dataset)
    label_pap_bow = "{}/processed/paper_bow/paper_bow.label.pkl".format(dataset)
    label_joined = "{}/processed/joined/joined.label.pkl".format(dataset)
    label_year = "{}/processed/paper_year/paper_year.label.pkl".format(dataset)
    
    created_files = [processed_author_ids, processed_acl_meta, relevant_paper_ids,
                     processed_paper_venue_network, processed_paper_author_network, processed_train_eval_test_dict,
                    processed_bow_count, processed_abstract_vecs, processed_title_vecs,
                    fin_pap_ven_feat, fin_pap_auth_feat, fin_pap_bow_feat, processed_paper_indices,
                    fin_pap_title_feat, fin_pap_joined_feat, fin_cossin_adj, adjacency_matrix]
    
#     process_metadata(raw_acl_meta, processed_acl_meta)
#     process_author_ids(raw_author_ids, processed_author_ids)
#     create_relevant_paper_id_list(processed_acl_meta, processed_author_ids, papers_text,
#                                   raw_acl, min_year, max_year, relevant_paper_ids, 10)
#     create_paper_author_network(processed_author_ids, processed_acl_meta, processed_paper_author_network)
#     create_paper_venue_network(processed_acl_meta, processed_paper_venue_network)
#     create_paper_graph(raw_acl, relevant_paper_ids, processed_paper_indices, adjacency_matrix)
#     create_train_val_test_dict(processed_acl_meta, processed_paper_indices, test_year, eval_year, processed_train_eval_test_dict) 

#     dump_paper_venue(400, processed_paper_indices, processed_paper_venue_network, fin_pap_ven_feat)
#     feature_to_label(fin_pap_ven_feat, adjacency_matrix, label_pap_ven)
#     dump_paper_author(processed_paper_indices, processed_paper_author_network, fin_pap_auth_feat)
#     feature_to_label(fin_pap_auth_feat, adjacency_matrix, label_pap_auth)
#     dump_paper_title(5000, processed_paper_indices, processed_acl_meta, stop_words, fin_pap_title_feat)
#     feature_to_label(fin_pap_title_feat, adjacency_matrix, label_pap_title, True, 1.0)
#    dump_paper_bow(5000, processed_paper_indices, stop_words, papers_text, fin_pap_bow_feat)
#     feature_to_label(fin_pap_bow_feat, adjacency_matrix, label_pap_bow)
#     abstract2vec(papers_text, processed_paper_indices, stop_words, processed_abstract_vecs)
#     title2vec(processed_acl_meta, processed_paper_indices, stop_words, processed_title_vecs)
#     dump_author_authors(adjacency_matrix, fin_pap_auth_feat, fin_pap_authAuth_feat)
#     doc_vec_to_adj(processed_title_vecs, "cos_sim_title.pkl")
#    dump_paper_year(processed_acl_meta, processed_paper_indices, fin_pap_year_feat)

#     join_feature_matrices(fin_pap_joined_feat, fin_pap_bow_feat, fin_pap_title_feat, fin_pap_auth_feat, fin_pap_year_feat)
    join_feature_matrices(fin_pap_joined_feat, fin_pap_joined_divi, fin_pap_title_feat, fin_pap_auth_feat)
#     feature_to_label(fin_pap_joined_feat, adjacency_matrix, label_joined)
#     dump_history(history_matrices, processed_paper_indices, processed_acl_meta, adjacency_matrix, fin_pap_auth_feat, [eval_year-10, eval_year-5, eval_year])


#     delete_files(created_files)
    print("End of Script.")
    
