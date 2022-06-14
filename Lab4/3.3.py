import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import datasets, tree
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
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
        #POCZATEK STANDARD SCALER
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_scaled = scaler.transform(X)
        #KONIEC STANDARD SCALER
        #POCZATEK GRP
        transformer = GaussianRandomProjection(n_components=15)
        transformer.fit(X_scaled[train_index])

        X_new = transformer.transform(X_scaled)
        #transformer.fit(X_scaled[test_index])
        #X_new_test = transformer.transform(X_train)
        #KONIEC GRP
        clf = cf
        clf.fit(X_new[train_index], y_train)
        pred = clf.predict(X_new[test_index])
        pkt.append(accuracy_score(y_test, pred))


    average.append(np.mean(pkt))
    sd.append(np.std(pkt))

table = [['','CART','kNN','SVC'],
         ["Srednia",(format(average[0],".3f")),(format(average[1],".3f")),(format(average[2],".3f"))],
         ["Odchylenie",(format(sd[0],".3f")),(format(sd[1],".3f")),(format(sd[2],".3f"))]]
print(tabulate(table))