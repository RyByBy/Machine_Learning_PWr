import matplotlib.pyplot as plt
from math import pi
import numpy as np
from tabulate import tabulate

scores = np.load("../../../../Downloads/results.npy")
scores = np.mean(scores, axis=1).T
#print(scores)
# metryki i metody
metrics=["Recall", 'Precision', 'Specificity', 'F1', 'G-mean', 'BAC']
methods=["RU", 'RO', 'SMOTE', 'EMPTY']
N = scores.shape[0]

# kat dla kazdej z osi
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# spider plot
ax = plt.subplot(111, polar=True)

# pierwsza os na gorze
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# po jednej osi na metryke
plt.xticks(angles[:-1], metrics)

# os y
ax.set_rlabel_position(0)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
color="grey", size=7)
plt.ylim(0,1)
# Dodajemy wlasciwe ploty dla kazdej z metod
test = []
srednia = []
for method_id, method in enumerate(methods):
    values=scores[:, method_id].tolist()
    #print(methods[method_id])
    #print(scores[:, method_id])
    test = scores[:, method_id]
    #print(np.mean(test))
    #srednia.append(np.mean(test))
    values += values[:1]
    # print(values)
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=method)
# Dodajemy legende
plt.legend(bbox_to_anchor=(1.15, -0.05), ncol=5)
# Zapisujemy wykres
plt.savefig("radar", dpi=200)
# print(test[0,0])
test = test.T
table = [['','RU','RO','SMOTE','EMPTY'],
         ["Recall",(format(test[0,0],".3f")),(format(test[1,0],".3f")),(format(test[2,0],".3f")),(format(test[3,0],".3f"))],
         ["Precision",(format(test[0,1],".3f")),(format(test[1,1],".3f")),(format(test[2,1],".3f")),(format(test[3,1],".3f"))],
         ["Specificity",(format(test[0,2],".3f")),(format(test[1,2],".3f")),(format(test[2,2],".3f")),(format(test[3,2],".3f"))],
         ["F1",(format(test[0,3],".3f")),(format(test[1,3],".3f")),(format(test[2,3],".3f")),(format(test[3,3],".3f"))],
         ["Gmean",(format(test[0,4],".3f")),(format(test[1,4],".3f")),(format(test[2,4],".3f")),(format(test[3,4],".3f"))],
         ["BalAcc",(format(test[0,5],".3f")),(format(test[1,5],".3f")),(format(test[2,5],".3f")),(format(test[3,5],".3f"))],
         ]
print(tabulate(table))