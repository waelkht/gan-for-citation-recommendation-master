import settings

from link_prediction import Link_pred_Runner

import os
import sys

if __name__ == '__main__':

    dataname = 'aan/joined/joined'
    
    try:
        dataname = sys.argv[1] # path
    except:
        print "No dataset given. Using: {}".format(dataname)
    print "Using: {}".format(dataname)
    
    model = 'arga_ae'  # 'arga_ae' or 'arga_vae'
    
    settings = settings.get_settings(dataname, model)
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    runner = Link_pred_Runner(settings)
    
    print "Training on: data/{}".format(dataname)
    #run
    runner.erun()
    
    print "Saved results in result/{}".format(dataname)
