from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
#from munkres import Munkres, print_matrix
import numpy as np

"""
This class stores the metrics. This is for the printing of the result.
"""
class linkpred_metrics():
    """
    Constructor:
    ##### INPUT #####
    :edges_pos   10% of the edges in adjacency Matrix But only uni-directional
    :edges_neg   as much edges as edges_pos; but they are fake and not from the adjacency matrix
    """
    def __init__(self, edges_pos, edges_neg):
        self.edges_pos = edges_pos
        self.edges_neg = edges_neg

    """
    This function calculates the ROC-Score
    ##### INPUTS #####
    :emb        The embeddings of the (V)AE
    :feas       A Dictionary:   [adj, num_features, num_nodes, features_nonzero, pos_weight, norm,
                                 adj_norm, adj_label, features, true_labels, train_edges, val_edges,
                                 val_edges_false, test_edges, test_edges_false, adj_orig]
    ##### RETURN ######
    :roc_score  The ROC-SCORE for the embedding
    :ap_score   The average_precision_score
    :emb        The embeddings
    """
    def get_roc_score(self, emb, feas):
        # if emb is None:
        #     feed_dict.update({placeholders['dropout']: 0})
        #     emb = sess.run(model.z_mean, feed_dict=feed_dict)

        """
        Just a sigmoid function. Works for scalars as well as matrices etc.
        ##### INPUTS #####
        :x        A scalar, vector or matrix of numbers
        ##### RETURN #####
        :y        The sigmoid function from x
        """
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        # The idea is the following: If two embeddings n_1 and n_2 have a similar vecs. v_1 and v_2
        # then the scalar-prod <v1, v2> is quite high. Thus a link between them is likely
        adj_rec = np.dot(emb, emb.T) # [N x 32] * [32 x N] = [N x N]
        
        preds = []
        pos = [] # NOT USED
        for e in self.edges_pos: # e is a coord. [x, y] of the adj matrix.
            preds.append(sigmoid(adj_rec[e[0], e[1]])) # Sigmoid maps the adj_rec[x, y] to a value in (0, 1)
            pos.append(feas['adj_orig'][e[0], e[1]]) # feas['adj_orig'][x, y] is always NOT 0 since we test against pos. edges.

        preds_neg = []
        neg = [] # NOT USED
        for e in self.edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]])) # Sigmoid maps the adj_rec[x, y] to a value in (-1, 1)
            neg.append(feas['adj_orig'][e[0], e[1]]) # feas['adj_orig'][x, y] is always 0 since we test against neg. edges.

        # [preds_1, preds_2, ...] hstack [pred_neg_1, pred_neg_2, ...] ==> [preds_1, preds_2, ..., pred_neg_1, pred_neg_2, ...]
        preds_all = np.hstack([preds, preds_neg])
        # preds_all = np.hstack([pos, neg]) would also work.
        # This is necessary, if the adjacency matrix does not only consist out of zeros and ones
        # labels_all = [1, 1, ..., 0, 0, ...]
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
        # get roc-score via sklearn
        roc_score = roc_auc_score(labels_all, preds_all)
        # get average_precision_score via sklearn
        ap_score = average_precision_score(labels_all, preds_all)

        # return the roc-score, the average_precision_score and the embeddings
        return roc_score, ap_score, emb


class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label


    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        print('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))

        fh = open('recoder.txt', 'a')

        fh.write('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore) )
        fh.write('\r\n')
        fh.flush()
        fh.close()

        return acc, nmi, adjscore

