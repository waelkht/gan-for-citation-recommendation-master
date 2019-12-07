import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
# from pandas.tools.plotting import parallel_coordinates

import numpy as np
import pickle as pkl
import random


"""
returns a random color
"""
def rand_color():
    r = lambda: random.randint(0,255)
    return '#%02X%02X%02X' % (r(),r(),r())

"""
This function visualizes an embedding of high dimensions.
It will show a plot from pyplot
##### input ####
:dataset The name of the dataset. e.g 'cora'
:model   The name of the model e.g. 'arga_ae'
"""
def vis(dataset, model):
    # own try: source https://www.apnorton.com/blog/2016/12/19/Visualizing-Multidimensional-Data-in-Python/
    emb = pkl.load(open('result/{}.embedding.{}'.format(dataset, model), 'r'))
    
    plt.scatter(emb[:,0], emb[:,1])
    plt.show()
    return
    
    label = pkl.load(open('result/{}.label.{}'.format(dataset, model), 'r'))
    emb_norm = (emb - np.min(emb))/(np.max(emb) - np.min(emb))
    k = max(label)+1 # the number of classes
    colors = list()
    for i in range(k):
        colors.append(rand_color())
    
    lda = LDA(n_components=2) # #2-dimensional LDA
    lda_transformed = pd.DataFrame(lda.fit_transform(emb_norm, label))
    # Plot all three series
    for i in range(k):
        plt.scatter(lda_transformed[label==i][0], lda_transformed[label==i][1], label="C{}".format(i), c=colors[i], s=5.0)
     
    # Display legend and show plot
    plt.legend(loc=3)
    plt.show()

#     pca = sklearnPCA(n_components=2) #2-dimensional PCA
#     transformed = pd.DataFrame(pca.fit_transform(emb_norm))
#     for i in range(k):
#         plt.scatter(transformed[label==i][0], transformed[label==i][1], label='Class 1', c=colors[i])
#         
#     plt.legend()
#     plt.show()
  
if __name__ == '__main__':
    vis('aan_paper_venue', 'arga_ae')