import matplotlib.pyplot as plt
import strlearn as sl
from sklearn.naive_bayes import GaussianNB
from skmultiflow.trees import HoeffdingTreeClassifier
from strlearn.utils import scores_to_cummean
from AdaBoost import AdaBoost
import numpy as np
from sklearn.metrics import accuracy_score
from strlearn.metrics import precision
from tabulate import tabulate

stream = sl.streams.StreamGenerator(n_chunks=400,
                                    chunk_size=100,
                                    n_classes=2,
                                    n_drifts=3,
                                    n_features=10)

clfs = [
    sl.ensembles.SEA(GaussianNB(), n_estimators=10),
    AdaBoost(),
    HoeffdingTreeClassifier(),
]
clf_names = [
    "SEA",
    "Ada",
    "HTC",
]

# Wybrana metryka
metrics = [accuracy_score,
           precision]
# Nazwy metryk
metrics_names = ["Acc Score",
                 "Precision"]

evaluator = sl.evaluators.TestThenTrain(metrics)
evaluator.process(stream,clfs)
srednia = []
odchylenie = []
test = []
tescik = []
ehh = []
#changed = []
fig, ax = plt.subplots(1, len(metrics), figsize=(24, 8))
fig, ax2 = plt.subplots(1, len(metrics), figsize=(24, 8))

scores_cm = scores_to_cummean(evaluator.scores)
print(scores_cm.shape)
for m, metric in enumerate(metrics):
    ax[m].set_title(metrics_names[m])
    ax2[m].set_title(metrics_names[m])
    ax[m].set_ylim(0, 1)
    ax2[m].set_ylim(0, 1)
    #changed[m].set_ylim(0, 1)
    for i, clf in enumerate(clfs):
        ax[m].plot(scores_cm[i,:,m], label=clf_names[i])
        ax2[m].plot(evaluator.scores[i,:,m], label=clf_names[i])
        srednia.append(np.mean(evaluator.scores[i, :, m]))
        odchylenie.append(np.std(evaluator.scores[i, :, m]))

    plt.ylabel("Metric")
    plt.xlabel("Chunk")
    ax[m].legend()
    ax2[m].legend()
table1 = [[metrics_names[0],'SEA','ADA','HTC'],
         ["Srednia",(format(srednia[0],".3f")),(format(srednia[1],".3f")),(format(srednia[2],".3f"))],
         ["Odchylenie",(format(odchylenie[0],".3f")),(format(odchylenie[1],".3f")),(format(odchylenie[2],".3f"))]]
table2 = [[metrics_names[1],'SEA','ADA','HTC'],
         ["Srednia",(format(srednia[3],".3f")),(format(srednia[4],".3f")),(format(srednia[5],".3f"))],
         ["Odchylenie",(format(odchylenie[3],".3f")),(format(odchylenie[4],".3f")),(format(odchylenie[5],".3f"))]]
print(tabulate(table1))
print(tabulate(table2))
plt.show()