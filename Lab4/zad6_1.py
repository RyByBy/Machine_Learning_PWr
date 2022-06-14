from sklearn.base import clone
from sklearn.base import ClassifierMixin, BaseEstimator


class SamplingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, base_preprocesing=None):
        self.base_estimator = base_estimator
        self.base_preprocesing = base_preprocesing

    def fit(self, X, y):
        if self.base_preprocesing != None:
            preproc = clone(self.base_preprocesing)
            X_new, y_new = preproc.fit_resample(X, y)
            self.clf = clone(self.base_estimator)
            self.clf.fit(X_new, y_new)
            return self
        else:
            self.clf = clone(self.base_estimator)
            self.clf.fit(X, y)
            return self

    def predict(self, X):
        return self.clf.predict(X)
