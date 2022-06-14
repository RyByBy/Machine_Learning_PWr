import numpy as np
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import datasets, tree
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from numpy.random import normal
from tabulate import tabulate

#3.1

X, y = datasets.make_classification(
            n_samples=500,
            n_features=30,
            n_informative=5,
            random_state=123
        )

mu, sigma = 0.2, 0.1
vec = normal(mu, sigma, X.shape[1])
X = X * vec
CART = tree.DecisionTreeClassifier()
kNN = KNeighborsClassifier()
SVC = SVC()

classifiers = [
    CART,
    kNN,
    SVC,
]

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=123)

average = []
sd = []

for cf_cnt, cf in enumerate(classifiers):
    pkt = []
    for train_index, test_index in rskf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #POCZATEK
        perc = SelectPercentile()
        perc.fit(X_train, y_train)
        X_new = perc.transform(X_train)
        X_new_test = perc.transform(X_test)
        #KONIEC
        clf = cf
        clf.fit(X_new, y_train)
        pred = clf.predict(X_new_test)
        pkt.append(accuracy_score(y_test, pred))


    average.append(np.mean(pkt))
    sd.append(np.std(pkt))

table = [['','CART','kNN','SVC'],
         ["Srednia",(format(average[0],".3f")),(format(average[1],".3f")),(format(average[2],".3f"))],
         ["Odchylenie",(format(sd[0],".3f")),(format(sd[1],".3f")),(format(sd[2],".3f"))]]
print(tabulate(table))