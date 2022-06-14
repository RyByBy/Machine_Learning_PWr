from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs
from sklearn.base import clone
from sklearn import neighbors
from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from classificator import Classifier
from sasiad import NearestNeighborClassifier
from tabulate import tabulate


#ZAD 3

X, y = make_classification(n_samples=200, n_features=2,n_redundant=0, random_state=123)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=123)

clf = Classifier()
clf.fit(X_train,y_train)

pred = clf.predict(X_test)
print ("Percentage: %.3f" % accuracy_score(y_test, pred))

#ZAD 2

clf = NearestNeighborClassifier()
clf.fit(X_train,y_train)

pred = clf.predict(X_test)
print ("My Neighbour: %.3f" % accuracy_score(y_test, pred))

nb=neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='brute')
nb.fit(X_train, y_train)
pred = nb.predict(X_test)
print("Neighbour sklearn %.3f" % accuracy_score(y_test, pred))


#ZAD 3
print("\n\n\n\n")

datasets2 = [
    make_moons(random_state=123),
    make_circles(random_state=123),
    make_blobs(random_state=123),
]
classifiers = [
    Classifier(),
    NearestNeighborClassifier(),
]

rskf = RepeatedStratifiedKFold(n_splits=5,n_repeats=2)
rskf.get_n_splits(X, y)


srednia =[]
odchylenie =[]
for ds_cnt, ds in enumerate(datasets2):
    X, y = ds


    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=123)
    for cf_cnt, cf in enumerate(classifiers):
        pkt = []
        for train_index, test_index in rskf.split(X,y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = clone(cf)
            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
            pkt.append(accuracy_score(y_test, pred))

        print(pkt)

        srednia.append(np.mean(pkt))
        odchylenie.append(np.std(pkt))

        # print("Wartosc srednia: %.3f" % srednia)
        # print("Wartosc odchylenia: %.3f" % odchylenie)

table = [['MOONS','CIRCLES','BLOBS'],
         [srednia[0],srednia[1],srednia[2]],
         [odchylenie[0],odchylenie[1],odchylenie[2]]]
print(tabulate(table))

