import settings

from clustering import Clustering_Runner # own code
from link_prediction import Link_pred_Runner # own code
from visualize import vis

import os
import sys
import tensorflow as tf

if __name__ == '__main__':

    dataname = 'aan/joined/joined'

    try:
        dataname = sys.argv[1] # path
    except:
        print "No dataset given. Using: {}".format(dataname)
    print "Using: {}".format(dataname)

    model = 'arga_ae'          # 'arga_ae' or 'arga_vae'
    task = 'link_prediction'         # 'clustering' or 'link_prediction'

    settings = settings.get_settings(dataname, model, task)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if task == 'clustering':
        runner = Clustering_Runner(settings)
    else:
        runner = Link_pred_Runner(settings)
    
    print "Training on: data/{}".format(dataname)
    #run
    runner.erun()
    
    print "Saved results in result/{}".format(dataname)
    #visualize
    # vis(dataname, model)
