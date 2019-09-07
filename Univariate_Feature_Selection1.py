'''
单变量特征选择
mutual_info_classif
（互信息）
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile,  mutual_info_classif


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# The iris dataset
iris = datasets.load_iris()
# Some noisy data not correlated
E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))
# Add the noisy data to the informative features
X = np.hstack((iris.data, E))
y = iris.target

X_indices = np.arange(X.shape[-1])
# 单变量mutual_info_classif选择
selector = SelectPercentile(mutual_info_classif, percentile=10)
selector.fit(X, y)
# 互信息MI的值（不仅可以获取互信息值，还可以自动选择好变量建立模型）
mutual_info = selector.scores_  # 等效 mutual_info = mutual_info_classif(X, y, discrete_features=False)
mutual_info /= mutual_info.max()
plt.bar(X_indices, mutual_info, width=.2, label='互信息', color='navy')
plt.title("# mutual_info_classif 变量选择")
plt.show()

