'''
案例一：
基于 Tree（树）的特征选取

案例二：
 14.4 Feature importances with forests of trees
 在合成数据上恢复有用特征的示例。
'''
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris, make_classification
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

iris = load_iris()
X, y = iris.data, iris.target
X_indices = np.arange(X.shape[-1])

# 随机森林
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
feature_importances = clf.feature_importances_
feature_importances /= feature_importances.max()
# 特征p值
plt.bar(X_indices - .45, feature_importances, width=.2)
plt.title("# ExtraTreesClassifier 变量选择")
# plt.show()




'''
 14.4 Feature importances with forests of trees
 在合成数据上恢复有用特征的示例。
'''
# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000,n_features=10,n_informative=3,n_redundant=0,n_repeated=0,n_classes=2,random_state=0,shuffle=False)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
#avg = np.mean([tree.feature_importances_ for tree in forest.estimators_], axis=0)  # 和上面的等效
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)  # 每棵树的变量特征的方差

indices = np.argsort(importances)[::-1]   # 对importances进行倒叙排序

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")

plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
