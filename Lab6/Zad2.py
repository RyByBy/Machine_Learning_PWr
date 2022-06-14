from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from Zad2Bag import BaggingClassifier2
import glob
import numpy as np
from scipy.stats import ttest_ind

Bagging_Weighted = BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=123),
                                            hard_voting=False,
                                            weight_mode=True,
                                            random_state=123)
Bagging_Clean = BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=123),
                                        hard_voting=False,
                                        weight_mode=False,
                                        random_state=123)
Bagging_HardVote_Weighted = BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=123),
                                                    hard_voting=True,
                                                    weight_mode=True,
                                                    random_state=123)
Bagging_Hardvote = BaggingClassifier2(base_estimator=DecisionTreeClassifier(random_state=123),
                                            hard_voting=True,
                                            weight_mode=False,
                                            random_state=123)

classifiers = [
    Bagging_Weighted,
    Bagging_Clean,
    Bagging_HardVote_Weighted,
    Bagging_Hardvote
]


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
print("\nBagging Weighted Mean scores:\n", format((mean_scores[0,0]),"4f"))
print("\nBagging Clean Mean scores:\n",format((mean_scores[0,1]),"4f"))
print("\nBagging Hardvote Weighted Mean scores:\n", format((mean_scores[0,2]),"4f"))
print("\nBagging Hardvote Mean scores:\n", format((mean_scores[0,3]),"4f"))


t_stat = np.zeros(shape=(len(classifiers), len(classifiers)), dtype=float)
p_val = np.zeros(shape=(len(classifiers), len(classifiers)), dtype=float)
adv = np.zeros(shape=(len(classifiers), len(classifiers)), dtype=float)
sig = np.zeros(shape=(len(classifiers), len(classifiers)), dtype=float)
s_better = np.zeros(shape=(len(classifiers), len(classifiers)), dtype=float)
alpha = 0.05

scores = scores[0].T

for i in range(len(classifiers)):
    for j in range(len(classifiers)):
        t_stat[i,j], p_val[i,j] = ttest_ind(scores[i], scores[j])

adv[t_stat > 0] = 1
sig[p_val <= alpha] = 1
s_better = adv*sig

print(f"t-stat:\n{t_stat}\np-value:\n{p_val}\nadvantage:\n{adv}\nsignificance:\n{sig}\nstat-better:\n{s_better}")