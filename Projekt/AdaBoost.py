import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.utils import check_X_y
from skmultiflow.utils.utils import get_dimensions
from sklearn.base import ClassifierMixin, clone
from sklearn.naive_bayes import GaussianNB

def compute_error(y, y_pred, w_i):
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int))) / sum(w_i)


def compute_alpha(error):
    return np.log((1 - error) / error)


def update_weights(w_i, alpha, y, y_pred):
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))

class AdaBoost:

    def __init__(self,base_clf=GaussianNB()):
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []
        self.classes = []
        self.data_window = []
        self.base_clf = base_clf

    def fit(self, X, y, M=100):
        self.alphas = []
        self.training_errors = []
        self.M = M

        for m in range(0, M):

            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)
            else:
                w_i = update_weights(w_i, alpha_m, y, y_pred)

            G_m = DecisionTreeClassifier(max_depth=1)
            G_m.fit(X, y, sample_weight=w_i)
            y_pred = G_m.predict(X)

            self.G_M.append(G_m)

            error_m = compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)

            alpha_m = compute_alpha(error_m)
            self.alphas.append(alpha_m)

        assert len(self.G_M) == len(self.alphas)
    def partial_fit(self, X, y, classes=None, M=100):
    #No remember function for project
        X, y = check_X_y(X, y)
        for m in range(0, M):

            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)
            else:
                w_i = update_weights(w_i, 0, y, y_pred)

            G_m = DecisionTreeClassifier(max_depth=1)
            G_m.fit(X, y, sample_weight=w_i)
            y_pred = G_m.predict(X)

            self.G_M.append(G_m)

            error_m = compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)

        self.alphas = []
        self.training_errors = []
        self.M = M

        return self
    def predict(self, X):

        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.M))

        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X)
            weak_preds.iloc[:,m] = y_pred_m

        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)

        return y_pred
