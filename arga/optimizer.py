import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    """
    Inits an optimzer
    #### INPUTS ####
    :preds                          The Reconstructions of the AE-Model [N^2]
    :labels                         orig_adj reshaped as [N^2]
    :pos_weight                     Position weight
    :norm                           The norm
    :d_real                         The output of the Discriminator which is NOT linked to the AE
    :d_fake                         The output of the Discriminator which is linked to the AE
    #### CREATED ####
    :self_discriminator_optimizer   for the Discriminator
    :self_generator_optimizer       for the Generator with penalty on bad embeddings
    :self_opt_op                    for the Generator without penalty on bad embeddings
    :self_grads_vars                the computed gradients of the self_optimizer
    """
    def __init__(self, preds, labels, pos_weight, norm, d_real, d_fake):
        preds_sub = preds
        labels_sub = labels

        self.real = d_real

        ##### Discrimminator Loss #####
        # Loss for the Discriminator - real
        self.dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.real), logits=self.real,name='dclreal'))
        # Loss for the Discriminator - fake
        self.dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake,name='dcfake'))
        # loss_total = loss_real + loss_fake
        self.discriminator_loss = self.dc_loss_fake + self.dc_loss_real

        ##### Generator loss ####
        generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake, name='gl'))

        # We calculate the cost: The positive examples get an extra weight W. The negative have a weight of 1
        # At the end the cost has to be normalized.
        
        ##### Encoder loss #####
        self.encoder_loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.generator_loss = generator_loss + self.encoder_loss
       
        all_variables = tf.trainable_variables()
        # get all the variables in the Generator: gen_den1
        # gen_var = [var for var in all_variables if 'gen_' in var.name] 
        # get all the variables in the Discriminator: dc_den1, dc_den2
        dc_var = [var for var in all_variables if 'dc_' in var.name] 
        # get all the variables in the (V)AE: e_dense_1, e_dense_2, (e_dense_3 for VAE)
        en_var = [var for var in all_variables if 'e_' in var.name]

      
        with tf.variable_scope(tf.get_variable_scope()):
            # minimizer the loss of the Discriminator        
            self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                             beta1=0.9, name='adam1').minimize(self.discriminator_loss, var_list=dc_var)
            # minimize the loss of the Generator
            self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                         beta1=0.9, name='adam2').minimize(self.generator_loss, var_list=en_var)
            # minimizer the loss of the Encoder
            self.encoder_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                         beta1=0.9, name='adam3').minimize(self.encoder_loss, var_list=en_var)
            


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, d_real, d_fake):
        preds_sub = preds
        labels_sub = labels

        # Discrimminator Loss
        dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
        dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
        self.discriminator_loss = dc_loss_fake + dc_loss_real

        # Generator loss
        self.generator_loss = tf.reduce_mean(
             tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

        # Encoder loss
        self.encoder_loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        # Latent loss
        self.log_lik = self.encoder_loss
        # = 1/(2N) * [N](1+2*\sigma - \my^2 - e^2\sigma)
        # This is the Kullback Leibniz difference.
        # KL(f_1, f_2) where f1 is a normal Distribution N(0,1) and f_2 is the normal distribution for the embeddings.
        # The closer f_2 gets to f_1 the smaller gets the loss
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                                                                   tf.square(tf.exp(model.z_log_std)), 1))
        # NOTE: self.kl is negative => self.cost += -(self.kl) <==> self.cost -= self.kl
        self.encoder_loss -= self.kl

        all_variables = tf.trainable_variables()
        # gen_var = [var for var in all_variables if 'gen_' in var.name] 
        dc_var = [var for var in all_variables if 'dc_' in var.op.name]
        en_var = [var for var in all_variables if 'e_' in var.op.name]


        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                         beta1=0.9, name='adam1').minimize(self.discriminator_loss, var_list=dc_var)
            self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                         beta1=0.9, name='adam2').minimize(self.generator_loss, var_list=en_var)
            self.encoder_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                         beta1=0.9, name='adam3').minimize(self.encoder_loss, var_list=en_var)
