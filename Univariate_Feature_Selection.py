'''
单变量特征选择
f_classif
（F检验）
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# The iris dataset
iris = datasets.load_iris()
# Some noisy data not correlated
E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))
# Add the noisy data to the informative features
X = np.hstack((iris.data, E))
y = iris.target
plt.figure(1)
plt.clf()
X_indices = np.arange(X.shape[-1])

# 单变量F-test选择
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
# 特征p值
plt.bar(X_indices - .45, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)', color='darkorange')

plt.title("# 单变量F-test变量选择")


# Compare to the weights of an SVM
# 未选择变量建立模型
plt.figure(2)
clf = svm.SVC(kernel='linear')
clf.fit(X, y)
svm_weights = (clf.coef_ ** 2).sum(axis=0)
svm_weights /= svm_weights.max()
plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight', color='navy')

# 根据选择的变量建立模型
clf_selected = svm.SVC(kernel='linear')
clf_selected.fit(selector.transform(X), y)   # 根据选择的变量建立模型
svm_weights_selected = (clf_selected.coef_ ** 2).sum(axis=0)
svm_weights_selected /= svm_weights_selected.max()
plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected, width=.2, label='SVM weights after selection', color='c')
plt.title("Comparing feature selection")
plt.xlabel('Feature number')
plt.yticks(())
plt.axis('tight')
plt.legend(loc='upper right')
plt.show()
