'''
交叉验证自动获取最佳特征数。
Recursive feature elimination
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification, load_iris

# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=8,
                           n_clusters_per_class=1, random_state=0)

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X, y)

# 输出结果
print("Optimal number of features : %d" % rfecv.n_features_)
# Plot number of features VS. cross-validation scores

# 最优特征的序号
support = np.argwhere(rfecv.support_==True)+1
print("Optimal index of features : {}".format(support))

plt.figure()
plt.subplot(1, 2, 1)
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)


# 案例2
iris = load_iris()
X2 = iris.data
y2 = iris.target
rfecv2 = RFECV(estimator=svc, step=1, cv=3)
rfecv2.fit(X2, y2)
print("iris data Optimal number of features : %d" % rfecv2.n_features_)
plt.subplot(1, 2, 2)
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.title('iris data example')
plt.plot(range(1, len(rfecv2.grid_scores_) + 1), rfecv2.grid_scores_)
support1 = np.argwhere(rfecv2.support_==True)+1
print("Optimal index of features : {}".format(support1))

plt.show()










