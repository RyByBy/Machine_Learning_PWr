from Zad3Bag  import RandomSubspaceEnsemble
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from Zad2Bag import BaggingClassifier2
import glob
from sklearn.tree import DecisionTreeClassifier

BAG5 = BaggingClassifier2(base_estimator=GaussianNB(),
                         n_estimators=5,
                         hard_voting=False,
                         random_state=123)

BAG10 = BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=123),
                        n_estimators=10,
                        hard_voting=False,
                        random_state=123)

BAG15 = BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=123),
                        n_estimators=15,
                        hard_voting=False,
                        random_state=123)
RSE5 = RandomSubspaceEnsemble(base_estimator=GaussianNB(),
                              n_estimators=5,
                              n_subspace_features=5,
                              hard_voting=False,
                              random_state=123)
RSE10 = RandomSubspaceEnsemble(base_estimator=GaussianNB(),
                              n_estimators=10,
                              n_subspace_features=5,
                              hard_voting=False,
                              random_state=123)
RSE15 = RandomSubspaceEnsemble(base_estimator=GaussianNB(),
                              n_estimators=15,
                              n_subspace_features=5,
                              hard_voting=False,
                              random_state=123)

classifiers = [
    BAG5,
    BAG10,
    BAG15,
    RSE5,
    RSE10,
    RSE15,
]
names = {
"Bagging classifier 5 estimators",
"Bagging classifier 10 estimators",
"Bagging classifier 15 estimators",
"RandomSubspaceEnsembl 5 estimators",
"RandomSubspaceEnsemble 10 estimators",
"RandomSubspaceEnsemble 15 estimators",
}
dataset = 'ring'
dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)

print("Total number of features", X.shape[1])




n_splits = 5
n_repeats = 10
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

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


#print("\nScores:\n", scores.shape)
#print(classifiers[1])
mean_scores = np.mean(scores, axis=2).T

#for cf_cnt, cf in enumerate(classifiers):
#    print("\nMean scores:\n", format((mean_scores[0,cf_cnt]),"4f"))


print(f"Bagging classifier 5 estimators:\n{mean_scores[0,0]}\nRandomSubspaceEnsemble 5 estimators\n{mean_scores[0,3]}")
print(f"Bagging classifier 10 estimators:\n{mean_scores[0,1]}\nRandomSubspaceEnsemble 10 estimators\n{mean_scores[0,4]}")
print(f"Bagging classifier 15 estimators:\n{mean_scores[0,2]}\nRandomSubspaceEnsemble 15 estimators\n{mean_scores[0,5]}")