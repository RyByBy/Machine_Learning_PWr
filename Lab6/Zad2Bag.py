import numpy as np
from sklearn.ensemble import BaggingClassifier, BaseEnsemble
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from scipy.stats import mode


class BaggingClassifier2(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=5, random_state=None, hard_voting=True,
                 weight_mode=False):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.hard_voting = hard_voting
        self.weight_mode = weight_mode
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features = X.shape[1]
        self.weights = np.zeros(self.n_estimators)

        self.ensemble_ = []
        for i in range(self.n_estimators):
            self.bootstrap = np.random.choice(len(X), size=len(X), replace=True)
            self.ensemble_.append(clone(self.base_estimator).fit(X[self.bootstrap], y[self.bootstrap]))
            self.weights[i] = accuracy_score(self.ensemble_[i].predict(X), y)
        return self

    def predict(self, X):
        check_is_fitted(self, "classes_")
        X = check_array(X)

        if self.hard_voting:
            pred_ = []

            if self.weight_mode == True:
                for i, clfs in enumerate(self.ensemble_):
                    pred_.append(clfs.predict(X))
                    pred_[i] = pred_[i] * self.weights[i]

                pred_ = np.array(pred_)
                prediction = mode(pred_, axis=0)[0].flatten()
                prediction = np.around(prediction).astype(int)
                return self.classes_[prediction]


            else:
                for i, clfs in enumerate(self.ensemble_):
                    pred_.append(clfs.predict(X))

                pred_ = np.array(pred_)
                prediction = mode(pred_, axis=0)[0].flatten()
                return self.classes_[prediction]
        else:
            if self.weight_mode == False:
                esm = self.ensemble_support_matrix(X)
                average_support = np.mean(esm, axis=0)
                prediction = np.argmax(average_support, axis=1)
                return self.classes_[prediction]

            else:
                esm = self.ensemble_support_matrix(X)
                for i, clf in enumerate(self.ensemble_):
                    esm[i] = esm[i] * self.weights[i]
                average_support = np.mean(esm, axis=0)
                prediction = np.argmax(average_support, axis=1)
                return self.classes_[prediction]

    def ensemble_support_matrix(self, X):
        probas_ = []

        for i, clfs in enumerate(self.ensemble_):
            probas_.append(clfs.predict_proba(X))

        return np.array(probas_)

