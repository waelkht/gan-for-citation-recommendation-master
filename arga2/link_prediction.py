from __future__ import print_function
import os
from preprocessing import construct_feed_dict
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
from constructor import get_placeholder, get_model, format_data, get_optimizer, update
from metrics import linkpred_metrics
import output_data
import Z_recommend

flags = tf.app.flags
FLAGS = flags.FLAGS

class Link_pred_Runner():
    def __init__(self, settings):
        self.data_name = settings['data_name'] # eg. ''aan/paper_vec/paper_title_vec'
        self.iteration = settings['iterations'] # eg. 20
        self.model = settings['model'] # eg. arga_ae

    def erun(self):
        model_str = self.model # eg. 'arga_vae'
        # formatted data as dictionary ...
        # ... holds ['labels', 'num_features', 'num_nodes', 'pos_weight', 'norm',
        #            'features', 'train', 'val', 'val_false', 'test', 'test_false', 'features_nonzero', 'divisions']
        print("[INFO] Formatting Data")
        feas = format_data(self.data_name)

        # Define placeholders as dictionary of tf.placeholders(...)
        # ... holds ['features'[Nxm],'labels'[Nxn], dropout'(float), 'real_distribution'[Nx16]]
        print("[INFO] Getting placeholders")
        placeholders = get_placeholder(feas['num_nodes'])
        
        # x = feas['train'].toarray()
        # y = feas['col_weight']
        # import numpy as np
        # z = np.multiply(x, y)
        # np.sum(z, axis=0)

        # construct model
        # d_real:        A float indicating if the discriminator thinks the  input is real
        # discriminator: the object of the discriminator (not linked to the AE)
        # ae_model:      the ae_model
        print("[INFO] Getting model")
        d_real, discriminator, ae_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])

        # Optimizer
        # the optimizer now holds three variables:
        # opt.discriminator_optimizer   The optimizer for the Discriminator
        # opt.generator_optimizer       The optimizer for the Generator with penalty on bad embeddings
        # opt.optimizer                 The optimizer for the Generator without penalty on bad embeddings
        # opt.grads_vars                The computed gradients of the self_optimizer
        print("[INFO] Get Optimizers")
        opt = get_optimizer(model_str, ae_model, discriminator, placeholders, feas['pos_weight'], feas['col_weight'], d_real, feas['num_nodes'], feas['num_features'], feas['train_mask'])

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        print("[INFO] Training Model:")
        print("[INFO] ADJ-Size: ", feas['labels'].shape)
        print("[INFO] Features-Size: ", feas['features'].shape)
        print("[INFO] Training iterations: {}".format(FLAGS.iterations))
        
        # Train model
        for epoch in range(self.iteration):

            # Run the model, receive: embeddings, average_cost, discriminator- and generator loss
            rec, avg_cost, d_loss, g_loss = update(ae_model, opt, sess, feas['labels'], feas['features'], placeholders)

            # run test edges
            if epoch % 1 == 0:
                lm_val = linkpred_metrics(feas['val'], feas['val_false'], feas['val_mask'])
                recall_score = lm_val.get_roc_score(rec, feas)
                print("[EVAL] Epoch:", '%04d' % (epoch + 1),
                      "train_loss=", "{:.5f}".format(avg_cost),
                      "d_loss=", "{:.5f}".format(d_loss),
                      "g_loss=", "{:.5f}".format(g_loss),
                      "recall_score=", "{:.5f}".format(recall_score))
            
        print("[INFO] finished training. Saving results now...")
                
        # last but no least we want to get final reconstructions, with 0.0 dropout
        feed_dict = construct_feed_dict(feas['features'], feas['labels'], 0.0, placeholders)
        rec = sess.run(ae_model.reconstructions, feed_dict=feed_dict)
        
        # dump the outputs from the model
        train_eval_rec, test_rec, train_and_eval_features, test_features, adj_sort = output_data.dump(rec, feas['features'], self.data_name, self.model)
        
        # Evaluate the recommendations
        Z_recommend.perf(self.data_name, self.model,
                         divisions = feas['divisions'],
                         train_eval_rec = train_eval_rec,
                         test_rec = test_rec,
                         train_eval_features = train_and_eval_features,
                         test_features = test_features,
                         adj_sort = adj_sort)
        
        print("FINISHED!")
        
        