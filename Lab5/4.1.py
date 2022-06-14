import glob
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

GNB = GaussianNB()
kNN = KNeighborsClassifier()
CART = tree.DecisionTreeClassifier(random_state=123)

clsf = [
    GNB,
    kNN,
    CART,
]
n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=123)
average = []
sd = []
filenames = glob.glob('datasets/*.csv')
scores = np.zeros((len(clsf), len(glob.glob('datasets/*.csv')), n_splits * n_repeats))

for cf_cnt, cf in enumerate(clsf):
    for files_cnt, filename in enumerate(filenames):
        dataset = np.genfromtxt(filename,delimiter=',')
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)
        pkt = []
        for fold_cnt, (train_index, test_index) in enumerate(rskf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = cf
            clf.fit(X[train_index], y[train_index])
            pred = clf.predict(X[test_index])
            scores[cf_cnt, files_cnt, fold_cnt] = accuracy_score(y[test_index], pred)

np.save('results', scores)

scores = np.load('results.npy')
print("\nScores:\n", scores.shape)

mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", mean_scores)