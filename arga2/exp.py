import tensorflow as tf
import numpy as np

y_true = np.array([[0, 0, 1], [1, 0, 0]], dtype=np.float32)
y_pred = np.array([[0, 0, 100000], [100000, 0, 0]], dtype=np.float32)

def cross_entropy(y_true, y_pred):
    """
    This function calculates the cross entropy:
    Loss = -y*log(y') -(1-y)+(log(1-y'))
    ##### INPUTS #####
    :y_true   The True labels: y
    :y_pred   The predicted labels
    """
    loss =  tf.reduce_sum(-tf.multiply(y_true, tf.log(y_pred)) - tf.multiply((1. - y_true), tf.log(1. - y_pred)), axis=1)
    return loss

R = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred), axis=1)

with tf.Session() as sess:
    r = sess.run(R)

print r
