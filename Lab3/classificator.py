import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import euclidean_distances
from sklearn.base import ClassifierMixin, BaseEstimator

class Classifier(ClassifierMixin, BaseEstimator):

    def __init__(self) -> None:
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes, licznoscKlas = np.unique(y, return_counts=True)
        major_class = np.argmax(licznoscKlas,axis=0)
        self.X_, self.y_ = X, y
        self.classes = major_class
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]