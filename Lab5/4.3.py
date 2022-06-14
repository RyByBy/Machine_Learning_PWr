import numpy as np
from scipy.stats import rankdata
from scipy.stats import ranksums
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

scores = np.load('results.npy')

GNB = GaussianNB()
kNN = KNeighborsClassifier()
CART = tree.DecisionTreeClassifier(random_state=123)

clfs = [
    GNB,
    kNN,
    CART,
]


mean_scores = np.mean(scores, axis=2).T
ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)
print("\nRanks:\n", ranks)

alpha = .05
w_stat = np.zeros(shape=(len(clfs), len(clfs)), dtype=float)
p_val = np.zeros(shape=(len(clfs), len(clfs)), dtype=float)
adv = np.zeros(shape=(len(clfs), len(clfs)), dtype=float)
sig = np.zeros(shape=(len(clfs), len(clfs)), dtype=float)

s_better = np.zeros(shape=(len(clfs), len(clfs)), dtype=float)

for i in range(len(clfs)):
    for j in range(len(clfs)):
        w_stat[i, j], p_val[i, j] = ranksums(ranks.T[i], ranks.T[j])

print(w_stat)

adv[w_stat > 0] = 1
sig[p_val <= alpha] = 1
s_better = adv*sig

print(f"t-stat:\n{w_stat}\np-value:\n{p_val}\nadvantage:\n{adv}\nsignificance:\n{sig}\n")#stat-better:\n{s_better}")