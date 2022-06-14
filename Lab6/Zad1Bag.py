import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.base import clone
from scipy.stats import mode

BASE_MODEL = DecisionTreeClassifier()
ENSAMBLE_SIZE = 5


class BaggingClassifier1(ClassifierMixin, BaseEstimator):
    def __init__(self):
        self.clfs_ = None

    def fit(self, X, y):
        self.clfs_ = []

        for i in range(ENSAMBLE_SIZE):
            clf = clone(BASE_MODEL)
            bootstrap = np.random.choice(len(X), size=len(X), replace=True)
            clf.fit(X[bootstrap], y[bootstrap])
            self.clfs_.append(clf)

        return self

    def predict(self, X):
        predictions = []
        for clf in self.clfs_:
            predictions.append(clf.predict(X))
        predictions = np.array(predictions)
        return mode(predictions, axis=0)[0].flatten()

