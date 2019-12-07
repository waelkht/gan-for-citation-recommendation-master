import numpy as np
import Z_recommend


"""
This class stores the metrics. This is for the printing of the result.
"""
class linkpred_metrics():
    """
    Constructor:
    ##### INPUT #####
    :pos   Some true lables
    :neg   Some false labels
    """
    def __init__(self, pos, neg, mask):
        self.pos = pos
        self.neg = neg
        self.mask = mask

    """
    This function calculates the ROC-Score
    ##### INPUTS #####
    :emb        The embeddings of the (V)AE
    :feas       A Dictionary:   ['labels', 'num_features', 'num_nodes', 'pos_weight', 'norm',
                                'features', 'train', 'val', 'val_false', 'test', 'test_false', 'features_nonzero']
    ##### RETURN ######
    :roc_score  The ROC-SCORE for the embedding
    :ap_score   The average_precision_score
    :emb        The embeddings
    """
    def get_roc_score(self, rec, feas):

        # We calculate the mean of each column.
        MEAN = np.mean(rec[np.array(feas['train_mask']), :], axis = 0)

        mask = np.array(self.mask)
        
        reconstructions = rec[mask, :] - MEAN
        labels = feas['labels'][mask, :]
        
        best_columns = Z_recommend.get_most_top_n_indices(reconstructions)
        real_ones = Z_recommend.get_real_indices(labels)
        assert(len(best_columns) == len(real_ones))
        
        # just calculate the first 100
        avg = 0.
        for i in range(len(best_columns)):
            pred = best_columns[i]
            real = real_ones[i]
            common = set(pred).intersection(set(real))
            if len(real) > 0:
                avg += (len(common)*1.) / len(real)
        
        avg /= len(best_columns)
        
        return avg
        # len(set(best_columns[0]).intersection(best_columns[1]))
    
