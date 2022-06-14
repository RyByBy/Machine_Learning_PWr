from sklearn import datasets
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
import numpy as np
from sklearn.naive_bayes import  GaussianNB
from zad6_1 import SamplingClassifier

X, y = datasets.make_classification(
    n_samples=500,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    weights=[0.15,0.75],
    random_state=1
)
y = np.reshape(y, (X.shape[0],1)).astype(int)
dataset = np.concatenate((X,y), axis=1)
np.savetxt('datasets/data1.csv', dataset, delimiter=',')

X, y = datasets.make_classification(
    n_samples=500,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    weights=[0.01,0.99],
    random_state=1
)
y = np.reshape(y, (X.shape[0],1)).astype(int)
dataset = np.concatenate((X,y), axis=1)
np.savetxt('datasets/data2.csv', dataset, delimiter=',')

X, y = datasets.make_classification(
    n_samples=500,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    weights=[0.1,0.9],
    random_state=1,
    flip_y= 0.05
)
y = np.reshape(y, (X.shape[0],1)).astype(int)
dataset = np.concatenate((X,y), axis=1)
np.savetxt('datasets/data3.csv', dataset, delimiter=',')



X, y = datasets.make_classification(
    n_samples=500,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    weights=[0.1,0.5,0.5],
    random_state=1
)
y = np.reshape(y, (X.shape[0],1)).astype(int)
dataset = np.concatenate((X,y), axis=1)
np.savetxt('datasets/data4.csv', dataset, delimiter=',')

X, y = datasets.make_classification(
    n_samples=500,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    random_state=1
)
y = np.reshape(y, (X.shape[0],1)).astype(int)
dataset = np.concatenate((X,y), axis=1)
np.savetxt('datasets/data5.csv', dataset, delimiter=',')







clfs = {
    "RU" : SamplingClassifier(base_estimator=GaussianNB(), base_preprocesing=RandomUnderSampler(random_state=123)),
    "RO" : SamplingClassifier(base_estimator=GaussianNB(), base_preprocesing=RandomOverSampler(random_state=123)),
    "SMOTE": SamplingClassifier(base_estimator=GaussianNB(), base_preprocesing=SMOTE(random_state=123)),
    "Empty" : SamplingClassifier(base_estimator=GaussianNB()),
}

metrics = {
    "Recall": recall,
    "Precision": precision,
    "Specificity": specificity,
    "F1score": f1_score,
    "mean": geometric_mean_score_1,
    "Balanced Acc": balanced_accuracy_score,

}

datasets = [
    "data1",
    "data2",
    "data3",
    "data4",
    "data5",
]

n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
scores = np.zeros((len(datasets), len(clfs), n_splits * n_repeats, len(metrics)))


for data_id, data_name in enumerate(datasets):
        dataset = np.genfromtxt("datasets/%s.csv" % (data_name) , delimiter=',')
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
                for clf_id, clf_name in enumerate(clfs):
                    clf = clone(clfs[clf_name])
                    clf.fit(X[train], y[train])
                    y_pred = clf.predict(X[test])
                    for m_id, m_name in enumerate(metrics):
                        mtr = metrics[m_name]
                        scores[data_id, clf_id, fold_id, m_id] = mtr(y[test],y_pred)

np.save('results', scores)
print(scores)