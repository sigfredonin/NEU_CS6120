import numpy as np
from keras.callbacks import Callback

from sklearn.metrics import f1_score
from scipy.stats import pearsonr

class Metrics(Callback):

    def __init__(self, val_data, val_labels):
        self.val_data = val_data
        self.val_labels = val_labels

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_pearson_rs = []
        self.val_pearson_ps = []

    def on_epoch_end(self, epoch, logs={}):
        val_predicted = (np.asarray(self.model.predict(self.val_data))).round()
        _val_f1 = f1_score(self.val_labels, val_predicted)
        _val_pearson_r, _val_pearson_p = pearsonr(self.val_labels, val_predicted)
        self.val_f1s.append(_val_f1)
        self.val_pearson_rs.append(_val_pearson_r)
        self.val_pearson_ps.append(_val_pearson_p)
        print(" — val_f1: %f — val_pearson_r: %f — val_pearson_p: %f" % \
            (_val_f1, _val_pearson_r, _val_pearson_p))
