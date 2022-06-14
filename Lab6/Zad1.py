import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from Zad1Bag import BaggingClassifier1
from sklearn.tree import DecisionTreeClassifier
import glob

Bagging = BaggingClassifier1()
CART = DecisionTreeClassifier(random_state=1234)

classifiers = [
    Bagging,
    CART,
]

X, y = datasets.make_classification(n_samples=100,
                                    n_classes=2,
                                    n_informative=2,
                                    random_state=123)


n_splits = 5
n_repeats = 10
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=123)
average = []
sd = []
filenames = glob.glob('datasets/a*.csv')

scores = np.zeros((len(classifiers), len(glob.glob('datasets/a*')), n_splits * n_repeats))
for cf_cnt, cf in enumerate(classifiers):
    for ff_cnt, filename in enumerate(filenames):
        dataset = np.genfromtxt(filename,delimiter=',')
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)
        pkt = []
        for fold_cnt, (train_index, test_index) in enumerate(rskf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = cf
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            scores[cf_cnt, ff_cnt, fold_cnt] = accuracy_score(y_test, pred)

print("\nScores:\n", scores.shape)

mean_scores = np.mean(scores, axis=2).T
print("\nBagging Mean scores:\n", format((mean_scores[0,0]),"4f"))
print("\nCART Mean scores:\n",format((mean_scores[0,1]),"4f"))