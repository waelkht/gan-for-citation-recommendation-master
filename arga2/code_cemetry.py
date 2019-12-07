import pickle as pkl
import io

from sklearn.preprocessing import normalize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import scipy.sparse as sp
    
"""
output the real citation for a given index in the sorted adj. matrix
"""    
def get_orig_citation(idx):
    # <paper-id> ==> idx_adj
    paper_ind = pkl.load(open("../Datasets/AAN/processed/paper_indices.pkl", 'r'))
    # idx_adj ==> idx_sort
    new_ind = pkl.load(open("new_ind", 'r'))
    
    orig_pos_in_adj = [k for k, v in new_ind.items() if v == idx][0]
    orig_paper = [id for id, pos in paper_ind.items() if pos == orig_pos_in_adj][0]
    return orig_paper

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
    
    MAX = 5
    
    class MyCorpus():
        
        def __init__(self, dir_name):
            self.dir_name = dir_name
            
        def __iter__(self):
            i = 0
            for k, n in enumerate(nodes[0:MAX]):
                path = "{}/{}.txt".format(self.dir_name, n)
                try:
                    # open file and get abstract
                    f = io.open(path, 'r')
                    abstract = f.read()
                    f.close()
                    
                    # get a TaggedDocument based on the abstract
                    abstract = preprocess_text(abstract, stop_words)
                    tokens = abstract.split()
                    td = TaggedDocument(words=tokens, tags=[k])
                    
                    # yield the TaggedDocument
                    yield td
                except:
                    print("could not open {}".format(path))
                    pass
                i = i+1
                if i % 100 == 0:
                    print "[INFO] {} of {} papers".format(i, len(nodes))

    tagged_data = MyCorpus(directory_name)
    
    # TODO: do this with the pretrained Doc2Vec model
    
    max_epochs = 5
    vec_size = 150
    
    model = Doc2Vec(dm = 1, min_count=5, window=10, size=vec_size, sample=1e-4, negative=10)
      
    model.build_vocab(tagged_data)
    
    for _ in range(max_epochs):
        model.train(tagged_data, epochs = 10, total_examples=model.corpus_count)
        # decrease the learning rate
#         model.alpha -= 0.0002
        # fix the learning rate, no decay
#         model.min_alpha = model.alpha        print('Training iteration {0}'.format(epoch))

    model.save("abstract2vec-model.bin")
    print("[INFO] Model Saved")
    
    # get the embeddings
    embedding = sp.lil_matrix((len(nodes), vec_size))
    for i, n in enumerate(nodes[0:MAX]):
        embedding[paper_indices[n], :] = model.docvecs[i]
    
    embedding = normalize(embedding, norm='l1', axis=1)
    embedding = embedding.tocsr()
    
    pkl.dump(embedding, open(result_name, 'w'))
    print "[INFO] Papers 2 Vec finished"
    

    