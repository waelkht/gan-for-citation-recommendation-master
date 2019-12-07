from __future__ import division
from __future__ import print_function
import os
import numpy as np
from preprocessing import construct_feed_dict
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
from constructor import get_placeholder, get_model, format_data, get_optimizer, update
from metrics import linkpred_metrics
import output_data
import Z_recommend

class Link_pred_Runner():
    def __init__(self, settings):
        self.data_name = settings['data_name'] # eg. ''aan/paper_vec/paper_title_vec'
        self.iteration = settings['iterations'] # eg. 20
        self.model = settings['model'] # eg. arga_ae

    def erun(self):
        model_str = self.model # eg. 'arga_vae'
        # formatted data as dictionary ...
        # ... holds [adj, num_features, num_nodes, features_nonzero, pos_weight, norm, adj_norm, adj_label, features,
        #            true_labels, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, adj_orig]
        print("[INFO] Formatting Data")
        feas = format_data(self.data_name)

        # Define placeholders as dictionary of tf.placeholders(...)
        # ... holds ['features'[Nxm],'adj'[NxN], 'adj_orig'[NxN], 'dropout'(float), 'real_distribution'[Nx32]]
        print("[INFO] Getting placeholders")
        placeholders = get_placeholder(feas['adj'])

        # construct model
        # d_real: A float indicating if the discriminator thinks the  input is real
        # discriminator: the object of the discriminator (not linked to the AE)
        # ae_model: the ae_model
        print("[INFO] Getting model")
        d_real, discriminator, generator, ae_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])

        # Optimizer
        # the optimizer now holds three variables:
        # opt.discriminator_optimizer   The optimizer for the Discriminator
        # opt.generator_optimizer       The optimizer for the Generator with penalty on bad embeddings
        # opt.optimizer                 The optimizer for the Generator without penalty on bad embeddings
        # opt.grads_vars                The computed gradients of the self_optimizer
        print("[INFO] Get Optimizers")
        opt = get_optimizer(model_str, ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], d_real, feas['num_nodes'])

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        print("[INFO] Training Model:")
        print("[INFO] ADJ-Size: ", feas['adj'].shape)
        print("[INFO] Features-Size: ", feas['features'][2])
        # Train model
        for epoch in range(self.iteration): # [0:200]

            # Optimize the (V)AE, the Generator and the Discriminator.
            # emb = the embeddings [N x 32] and avg_cost  = cost for the (V)AE: (X^0 - X')
            emb, avg_cost, gen_loss = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])

            # run test edges
            lm_test = linkpred_metrics(feas['val_edges'], feas['val_edges_false'])
            roc_score, ap_score, _ = lm_test.get_roc_score(emb, feas)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost),
                  "val_roc=", "{:.5f}".format(roc_score),
                  "val_ap=", "{:.5f}".format(ap_score),
                  "gen_loss=", "{:.5f}".format(gen_loss))
            
        print("[INFO] Testing performance...")
              
        # last but no least we want to get final embeddings
        feed_dict = construct_feed_dict(feas['adj_norm'], feas['adj_label'], feas['features'], placeholders)
        feed_dict.update({placeholders['dropout']: 0}) 
        emb = sess.run(ae_model.z_mean, feed_dict=feed_dict)
  
        # dump the outputs from the model
        train_eval_embeddings, test_embeddings, adj_sort = output_data.dump(emb, feas["adj_orig"], self.data_name, self.model)
        # Evaluate the recommendations
        Z_recommend.perf(self.data_name, self.model,
                                 train_eval_emb = train_eval_embeddings,
                                 test_emb = test_embeddings,
                                 orig = adj_sort)
        print("FINISHED!")
        
        
