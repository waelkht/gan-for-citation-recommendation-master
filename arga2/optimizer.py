import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS

class OptimizerAE(object):
    """
    Inits an optimzer
    #### INPUTS ####
    :preds                          The Reconstructions of the AE-Model [N x N]
    :labels                         orig_adj reshaped as [N x N]
    :pos_weight                     Position weight
    :col_weight                     The weight of a column.
    :d_real                         The output of the Discriminator which is NOT linked to the AE
    :d_fake                         The output of the Discriminator which is linked to the AE
    :train_mask                     The train maks, indicating the nodes which belong to the training data
    #### CREATED ####
    :self_discriminator_optimizer   for the Discriminator
    :self_generator_optimizer       for the Generator with penalty on bad embeddings
    :self_opt_op                    for the Generator without penalty on bad embeddings
    :self_grads_vars                the computed gradients of the self_optimizer
    """
    def __init__(self, preds, labels, pos_weight, col_weight, d_real, d_fake, train_mask):
        
        preds = tf.boolean_mask(preds, train_mask)
        labels = tf.boolean_mask(labels, train_mask)

        ##### Discrimminator Loss #####        
        self.dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits = d_real, name='dclreal'))
        self.dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits = d_fake, name='dcfake'))
        self.dc_loss = self.dc_loss_fake + self.dc_loss_real

        ##### Generator loss ####
        generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(d_fake), logits = d_fake, name = 'gl'))
        
        ##### Reconstruction loss #####
        # self.cost = tf.reduce_mean(tf.reduce_sum(tf.multiply(col_weight, my_sigmoid_cross_entropy(y_pred=preds, y_true=labels, weight_true=pos_weight)), axis=1))
        # self.cost = tf.reduce_sum(my_sigmoid_cross_entropy(y_pred=preds, y_true=labels, col_weight=col_weight, weight_true=pos_weight))
        self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds, targets=labels, pos_weight=pos_weight))
        
        # we couple the generator loss with the encoder, since both want to improve the encoder
        # It prooved to be better.
        self.generator_loss = generator_loss + self.cost

        all_variables = tf.trainable_variables()
        # get all the variables in the Discriminator: dc_den1, dc_den2
        dc_var = [var for var in all_variables if 'dc_' in var.name] 
        # get all the variables in the (V)AE: e_dense_1, e_dense_2, (e_dense_3 for VAE)
        en_var = [var for var in all_variables if 'e_' in var.name]

      
        with tf.variable_scope(tf.get_variable_scope()):
            # minimizer the loss of the Discriminator        
            self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                             beta1=0.9, name='adam1').minimize(self.dc_loss, var_list=dc_var)
            # minimizer the loss of the Generator
            self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                         beta1=0.9, name='adam2').minimize(self.generator_loss, var_list=en_var)

        # minimizer for the loss of the (V)AE        
        # we use an adam-optimizer with the learning rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, d_real, d_fake):

        # Discrimminator Loss
        dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
        dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
        self.dc_loss = dc_loss_fake + dc_loss_real

        # Generator loss
        self.generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

        ##### Reconstruction loss #####
        self.cost = tf.reduce_mean(tf.reduce_sum(tf.losses.sigmoid_cross_entropy(multi_class_labels=preds, targets=labels), axis=1))
        
        all_variables = tf.trainable_variables()
        dc_var = [var for var in all_variables if 'dc_' in var.op.name]
        en_var = [var for var in all_variables if 'e_' in var.op.name]


        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                                  beta1=0.9, name='adam1').minimize(self.dc_loss, var_list=dc_var)

            self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                              beta1=0.9, name='adam2').minimize(self.generator_loss, var_list=en_var)
      
        # This is the Kullback Leibniz difference.
        # = 1/(2N) * [N](1+2*\sigma - \my^2 - e^2\sigma)
        # KL(f_1, f_2) where f1 is a normal Distribution N(0,1) and f_2 is the normal distribution for the embeddings.
        # The closer f_2 gets to f_1, the smaller gets the loss
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        # NOTE: self.kl is negative => self.cost += -(self.kl) <==> self.cost -= self.kl
        self.cost -= self.kl
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        
def my_sigmoid_cross_entropy(y_true, y_pred, col_weight, weight_true=1000.):
    """
    This function calculates the cross entropy:
    The sigmoid function is applied to the network output. ==> y' = sigmoid(y')
    Loss = -y*log(y') -(1-y)+(log(1-y'))
    ##### INPUTS #####
    :y_true       The True labels: y
    :y_pred       The predicted labels
    :col_weight   The weight for the columns
    :weight_true  The weight for the positive entries
    """
    
#     if col_weight.dtype != np.float32:
#         col_weight = col_weight.astype(np.float32)
#     
#     y_true = tf.multiply(col_weight, y_true)
    y_pred = tf.nn.sigmoid(y_pred)
    
    _loss_true = - tf.multiply(y_true, tf.log(y_pred +1e-7))# * weight_true
    _loss_false = - tf.multiply((1. - y_true), tf.log(1. - y_pred +1e-7))
    loss = _loss_true + _loss_false
    return loss

