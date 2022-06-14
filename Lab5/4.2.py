import numpy as np
from scipy.stats import ttest_ind
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

GNB = GaussianNB()
kNN = KNeighborsClassifier()
CART = tree.DecisionTreeClassifier(random_state=123)

clfs = [
    GNB,
    kNN,
    CART,
]
scores = np.load('results.npy')
scores = scores[0].T

alpha = .05

t_stat = np.zeros(shape=(len(clfs), len(clfs)), dtype=float)
p_val = np.zeros(shape=(len(clfs), len(clfs)), dtype=float)
adv = np.zeros(shape=(len(clfs), len(clfs)), dtype=float)
sig = np.zeros(shape=(len(clfs), len(clfs)), dtype=float)
s_better = np.zeros(shape=(len(clfs), len(clfs)), dtype=float)

for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_stat[i,j], p_val[i,j] = ttest_ind(scores[i], scores[j])

adv[t_stat > 0] = 1
sig[p_val <= alpha] = 1
s_better = adv*sig

print(f"t-stat:\n{t_stat}\np-value:\n{p_val}\nadvantage:\n{adv}\nsignificance:\n{sig}\nstat-better:\n{s_better}")
